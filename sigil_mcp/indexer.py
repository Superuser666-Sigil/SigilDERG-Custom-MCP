# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Hybrid code indexing for Sigil MCP Server.

Combines trigram-based text search with symbol extraction for IDE-like features.
Designed to work well with ChatGPT and other AI assistants via MCP.
"""

import sqlite3
import hashlib
import zlib
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Set, Callable, Sequence
import logging
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import threading
import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)

# Type alias for embedding function: takes sequence of texts, returns (N, dim) array
EmbeddingFn = Callable[[Sequence[str]], np.ndarray]


@dataclass
class Symbol:
    """Represents a code symbol (function, class, variable, etc.)."""
    name: str
    kind: str  # function, class, method, variable, etc.
    file_path: str
    line: int
    signature: Optional[str] = None
    scope: Optional[str] = None  # e.g., class name for methods


@dataclass
class SearchResult:
    """Represents a search result."""
    repo: str
    path: str
    line: int
    text: str
    doc_id: str
    symbol: Optional[Symbol] = None


class SigilIndex:
    """Hybrid index supporting both text and symbol search."""
    
    def __init__(
        self,
        index_path: Path,
        embed_fn: Optional[EmbeddingFn] = None,
        embed_model: str = "local"
    ):
        self.index_path = index_path
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.embed_fn = embed_fn
        self.embed_model = embed_model

        self.lance_db_path = self.index_path / "lancedb"
        self.lance = lancedb.connect(self.lance_db_path)
        self.vector_table_name = "code_vectors"

        # Global lock to serialize DB access across threads
        # (HTTP handlers + file watcher + vector indexing)
        self._lock = threading.RLock()
        
        # Use longer timeout for multi-process access (Admin API + main server)
        # 60 seconds should be enough for most operations
        self.repos_db = sqlite3.connect(
            self.index_path / "repos.db",
            check_same_thread=False,
            timeout=60.0  # 60 second timeout for database locks
        )
        # Enable WAL + sane defaults for concurrent readers / writers
        self.repos_db.execute("PRAGMA journal_mode=WAL;")
        self.repos_db.execute("PRAGMA synchronous=NORMAL;")
        self.repos_db.execute(
            "PRAGMA busy_timeout=60000;"
        )  # 60 seconds in milliseconds

        self.trigrams_db = sqlite3.connect(
            self.index_path / "trigrams.db",
            check_same_thread=False,
            timeout=60.0  # 60 second timeout for database locks
        )
        self.trigrams_db.execute("PRAGMA journal_mode=WAL;")
        self.trigrams_db.execute("PRAGMA synchronous=NORMAL;")
        self.trigrams_db.execute(
            "PRAGMA busy_timeout=60000;"
        )  # 60 seconds in milliseconds
        
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema."""
        # Repos and documents
        self.repos_db.execute("""
            CREATE TABLE IF NOT EXISTS repos (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                path TEXT,
                indexed_at TEXT
            )
        """)
        
        self.repos_db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                repo_id INTEGER,
                path TEXT,
                blob_sha TEXT UNIQUE,
                size INTEGER,
                language TEXT,
                FOREIGN KEY(repo_id) REFERENCES repos(id)
            )
        """)
        
        self.repos_db.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_path 
            ON documents(repo_id, path)
        """)
        
        # Symbol index for IDE-like features
        self.repos_db.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER,
                name TEXT,
                kind TEXT,
                line INTEGER,
                character INTEGER,
                signature TEXT,
                scope TEXT,
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            )
        """)
        
        self.repos_db.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_name 
            ON symbols(name)
        """)
        
        self.repos_db.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_kind 
            ON symbols(kind)
        """)
        
        # Embeddings table for semantic/vector search
        self.repos_db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                model TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(id),
                UNIQUE(doc_id, chunk_index, model)
            )
        """)
        
        self.repos_db.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_doc
            ON embeddings(doc_id)
        """)

        self.trigrams_db.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS file_content_fts 
            USING fts5(
                path UNINDEXED, 
                repo_id UNINDEXED, 
                content, 
                tokenize='trigram'
             )
        """
        )

        # Trigram inverted index for fast text search
        self.trigrams_db.execute("""
            CREATE TABLE IF NOT EXISTS trigrams (
                gram TEXT PRIMARY KEY,
                doc_ids BLOB
            )
        """)
        
        self.trigrams_db.execute("""
            CREATE INDEX IF NOT EXISTS idx_gram 
            ON trigrams(gram)
        """)
        
        self.repos_db.commit()
        self.trigrams_db.commit()
    
    def index_file(
        self,
        repo_name: str,
        repo_path: Path,
        file_path: Path,
    ) -> bool:
        """
        Re-index a single file (granular update).
        
        Args:
            repo_name: Logical repository name
            repo_path: Path to repository root
            file_path: Path to specific file to re-index
        
        Returns:
            True if file was indexed, False if skipped or error
        """
        with self._lock:
            try:
                # Get or create repo entry
                cursor = self.repos_db.cursor()
                cursor.execute(
                    "INSERT OR IGNORE INTO repos (name, path, indexed_at) "
                    "VALUES (?, ?, ?)",
                    (repo_name, str(repo_path), datetime.now().isoformat()),
                )
                cursor.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
                repo_id = cursor.fetchone()[0]
                
                # Determine language
                file_extensions = {
                    '.py': 'python', '.rs': 'rust', '.js': 'javascript',
                    '.ts': 'typescript', '.java': 'java', '.go': 'go',
                    '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.hpp': 'cpp',
                    '.rb': 'ruby', '.php': 'php', '.cs': 'csharp',
                    '.sh': 'shell', '.toml': 'toml', '.yaml': 'yaml',
                    '.yml': 'yaml', '.json': 'json', '.md': 'markdown',
                }
                ext = file_path.suffix.lower()
                language = file_extensions.get(ext, 'unknown')
                
                # Index the specific file
                result = self._index_file(
                    repo_id, repo_name, repo_path, file_path, language
                )
                
                if result:
                    # Rebuild trigrams for this file
                    self._update_trigrams_for_file(repo_id, repo_path, file_path)
                    self.repos_db.commit()
                    logger.info(f"Re-indexed {file_path.name} in {repo_name}")
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Error re-indexing {file_path}: {e}")
                return False
    
    def _update_trigrams_for_file(self, repo_id: int, repo_path: Path, file_path: Path):
        """Update trigrams for a specific file."""
        cursor = self.repos_db.cursor()
        
        # Calculate relative path (same way _index_file does)
        rel_path = file_path.relative_to(repo_path).as_posix()
        
        # Get document ID and blob SHA
        cursor.execute(
            "SELECT id, blob_sha FROM documents WHERE repo_id = ? AND path = ?",
            (repo_id, rel_path)
        )
        row = cursor.fetchone()
        if not row:
            return
        
        doc_id, blob_sha = row
        
        # Read file content
        content = self._read_blob(blob_sha)
        if not content:
            logger.debug(
                "_update_trigrams_for_file: blob %s not found for doc %s",
                blob_sha,
                doc_id,
            )
            return

        text = content.decode('utf-8', errors='replace').lower()
        logger.warning(
            "_update_trigrams_for_file: sample text start: %s",
            repr(text[:200]),
        )
        logger.warning(
            "_update_trigrams_for_file: contains 'new_function'? %s",
            'new_function' in text,
        )
        new_trigrams = self._extract_trigrams(text)
        logger.warning(
            "_update_trigrams_for_file: repo_id=%s doc_id=%s blob_sha=%s trigrams=%s",
            repo_id,
            doc_id,
            blob_sha,
            len(new_trigrams),
        )
        
        # Update trigrams database
        # Note: This is a simplified approach - for production, you'd want to
        # track which trigrams belong to which documents to enable removal
        for trigram in new_trigrams:
            cursor = self.trigrams_db.execute(
                "SELECT doc_ids FROM trigrams WHERE gram = ?", (trigram,)
            )
            row = cursor.fetchone()
            
            if row:
                # Add this doc_id if not already present
                try:
                    existing_ids = {
                        int(x) for x in zlib.decompress(row[0]).decode().split(',') if x
                    }
                except Exception:
                    # Defensive: log and reset existing ids if decompress/parse fails
                    logger.exception(
                        "Failed to parse existing doc_ids for trigram %s",
                        trigram,
                    )
                    existing_ids = set()
                existing_ids.add(doc_id)
            else:
                existing_ids = {doc_id}
            
            joined_ids = ','.join(str(doc_id) for doc_id in sorted(existing_ids))
            compressed = zlib.compress(joined_ids.encode())
            self.trigrams_db.execute(
                "INSERT OR REPLACE INTO trigrams (gram, doc_ids) VALUES (?, ?)",
                (trigram, compressed)
            )
        
        self.trigrams_db.commit()
        logger.warning(
            "_update_trigrams_for_file: committed %d trigrams for doc %s",
            len(new_trigrams),
            doc_id,
        )
        # Quick verification for common trigrams we expect from function names
        for sample in ("new", "fun", "unc"):
            c = self.trigrams_db.execute(
                "SELECT doc_ids FROM trigrams WHERE gram = ?",
                (sample,),
            )
            r = c.fetchone()
            if r:
                try:
                    ids = {
                        int(x)
                        for x in zlib.decompress(r[0]).decode().split(',')
                        if x
                    }
                except Exception:
                    ids = set()
                logger.warning(
                    "_update_trigrams_for_file: sample trigram %s -> %s",
                    sample,
                    sorted(ids),
                )
            else:
                logger.warning(
                    "_update_trigrams_for_file: sample trigram %s -> NOT FOUND",
                    sample,
                )
        # Dump first few grams present for quick inspection
        try:
            rows = list(self.trigrams_db.execute("SELECT gram FROM trigrams LIMIT 30"))
            grams = [r[0] for r in rows]
            logger.warning(
                "_update_trigrams_for_file: sample stored grams (first 30): %s",
                grams,
            )
        except Exception:
            logger.exception("_update_trigrams_for_file: failed to read stored grams")
    
    def index_repository(
        self,
        repo_name: str,
        repo_path: Path,
        force: bool = False
    ) -> dict[str, int]:
        """
        Index a repository for both text and symbol search.
        
        Args:
            repo_name: Logical repository name
            repo_path: Path to repository root
            force: If True, rebuild index even if up-to-date
        
        Returns:
            Statistics about indexing operation
        """
        with self._lock:
            logger.info(f"Indexing repository: {repo_name} at {repo_path}")
            
            start_time = datetime.now()
            stats: dict[str, int] = {
                "files_indexed": 0,
                "symbols_extracted": 0,
                "trigrams_built": 0,
                "bytes_indexed": 0
            }
            
            # Register or update repo
            cursor = self.repos_db.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO repos (name, path, indexed_at) "
                "VALUES (?, ?, ?)",
                (repo_name, str(repo_path), datetime.now().isoformat()),
            )
            repo_id = cursor.lastrowid
            if not repo_id:
                # Repo already exists, get its ID
                cursor.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
                repo_id = cursor.fetchone()[0]
            
            # Clear old trigram data if forcing rebuild
            if force:
                logger.info(f"Force rebuild: clearing old index data for {repo_name}")
                cursor.execute(
                    "DELETE FROM documents WHERE repo_id = ?", (repo_id,)
                )
                # Trigrams will be rebuilt entirely
                self.trigrams_db.execute("DELETE FROM trigrams")
            
            # Index all files
            file_extensions = {
                '.py': 'python', '.rs': 'rust', '.js': 'javascript',
                '.ts': 'typescript', '.java': 'java', '.go': 'go',
                '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.hpp': 'cpp',
                '.rb': 'ruby', '.php': 'php', '.cs': 'csharp',
                '.sh': 'shell', '.toml': 'toml', '.yaml': 'yaml',
                '.yml': 'yaml', '.json': 'json', '.md': 'markdown',
            }
            
            for file_path in repo_path.rglob("*"):
                if file_path.is_file() and not self._should_skip(file_path):
                    ext = file_path.suffix.lower()
                    language = file_extensions.get(ext, 'unknown')
                    
                    file_stats = self._index_file(
                        repo_id, repo_name, repo_path, file_path, language
                    )
                    if file_stats:
                        stats["files_indexed"] += 1
                        stats["symbols_extracted"] += file_stats.get("symbols", 0)
                        stats["bytes_indexed"] += file_stats.get("bytes", 0)
            
            self.repos_db.commit()
            
            # Build trigram index
            logger.info(f"Building trigram index for {repo_name}")
            trigram_count = self._build_trigram_index(repo_id)
            stats["trigrams_built"] = trigram_count
            
            elapsed = (datetime.now() - start_time).total_seconds()
            stats["duration_seconds"] = int(elapsed)
            
            logger.info(
                f"Indexed {repo_name}: {stats['files_indexed']} files, "
                f"{stats['symbols_extracted']} symbols, "
                f"{stats['trigrams_built']} trigrams in {elapsed:.1f}s"
            )
            
            return stats
    
    def _index_file(
        self,
        repo_id: int,
        repo_name: str,
        repo_root: Path,
        file_path: Path,
        language: str
    ) -> Optional[dict[str, int]]:
        """Index a single file."""
        try:
            content = file_path.read_bytes()
            blob_sha = hashlib.sha256(content).hexdigest()
            rel_path = file_path.relative_to(repo_root).as_posix()
            
            # Check if already indexed
            cursor = self.repos_db.cursor()
            cursor.execute(
                "SELECT id FROM documents WHERE blob_sha = ?", (blob_sha,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update path mapping if needed
                cursor.execute("""
                    UPDATE documents SET repo_id = ?, path = ?, language = ?
                    WHERE blob_sha = ?
                """, (repo_id, rel_path, language, blob_sha))
                return None  # Already indexed, skip
            
            # Store document metadata
            # Remove any existing document rows for this repo/path (stale entries)
            # This avoids returning an old doc_id that points to a previous blob_sha
            try:
                cursor.execute(
                    "SELECT id FROM documents WHERE repo_id = ? AND path = ?",
                    (repo_id, rel_path)
                )
                rows = cursor.fetchall()
                for (old_doc_id,) in rows:
                    # remove symbols referencing the old doc
                    cursor.execute(
                        "DELETE FROM symbols WHERE doc_id = ?",
                        (old_doc_id,),
                    )
                    # remove embeddings referencing the old doc (if any)
                    cursor.execute(
                        "DELETE FROM embeddings WHERE doc_id = ?",
                        (old_doc_id,),
                    )
                    # remove the old document row
                    cursor.execute("DELETE FROM documents WHERE id = ?", (old_doc_id,))
            except Exception:
                # Be defensive; if cleanup fails, continue (we'll still insert)
                logger.exception("Failed to cleanup old document rows for %s", rel_path)

            cursor.execute("""
                INSERT INTO documents (repo_id, path, blob_sha, size, language)
                VALUES (?, ?, ?, ?, ?)
            """, (repo_id, rel_path, blob_sha, len(content), language))
            doc_id = cursor.lastrowid
            
            # Store blob content (compressed)
            blob_dir = self.index_path / "blobs" / blob_sha[:2]
            blob_dir.mkdir(parents=True, exist_ok=True)
            blob_file = blob_dir / blob_sha[2:]
            if not blob_file.exists():
                blob_file.write_bytes(zlib.compress(content))
            
            # Extract symbols using ctags
            symbols = self._extract_symbols(file_path, language)
            for symbol in symbols:
                cursor.execute("""
                    INSERT INTO symbols (
                        doc_id, name, kind, line, character, signature, scope
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    symbol.name,
                    symbol.kind,
                    symbol.line,
                    0,  # character position
                    symbol.signature,
                    symbol.scope
                ))
            
            return {
                "symbols": len(symbols),
                "bytes": int(len(content))
            }
        
        except Exception as e:
            logger.warning(f"Error indexing {file_path}: {e}")
            return None
    
    def _extract_symbols(self, file_path: Path, language: str) -> List[Symbol]:
        """Extract symbols from a file using universal-ctags."""
        # Check if ctags is available
        try:
            result = subprocess.run(
                ["ctags", "--version"],
                capture_output=True,
                timeout=1
            )
            if result.returncode != 0:
                return []
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("ctags not available, skipping symbol extraction")
            return []
        
        try:
            # Run ctags with JSON output
            result = subprocess.run(
                [
                    "ctags",
                    "-f", "-",
                    "--output-format=json",
                    "--fields=+n+S+s",  # line number, signature, scope
                    str(file_path)
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return []
            
            symbols = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    if data.get("_type") == "tag":
                        symbols.append(Symbol(
                            name=data.get("name", ""),
                            kind=data.get("kind", "unknown"),
                            file_path=str(file_path),
                            line=data.get("line", 0),
                            signature=data.get("signature"),
                            scope=data.get("scope")
                        ))
                except json.JSONDecodeError:
                    continue
            
            return symbols
        
        except subprocess.TimeoutExpired:
            logger.warning(f"ctags timed out on {file_path}")
            return []
        except Exception as e:
            logger.debug(f"Error extracting symbols from {file_path}: {e}")
            return []
    
    def _build_trigram_index(self, repo_id: int) -> int:
        """Build trigram index for a repository's documents."""
        cursor = self.repos_db.cursor()
        trigram_map = {}  # gram -> set of doc_ids
        
        for doc_id, blob_sha in cursor.execute(
            "SELECT id, blob_sha FROM documents WHERE repo_id = ?",
            (repo_id,)
        ):
            content = self._read_blob(blob_sha)
            if content:
                text = content.decode('utf-8', errors='replace').lower()
                for trigram in self._extract_trigrams(text):
                    if trigram not in trigram_map:
                        trigram_map[trigram] = set()
                    trigram_map[trigram].add(doc_id)
        
        # Write to trigrams database
        for gram, doc_ids in trigram_map.items():
            # Get existing doc_ids if any
            cursor = self.trigrams_db.execute(
                "SELECT doc_ids FROM trigrams WHERE gram = ?", (gram,)
            )
            row = cursor.fetchone()
            
            if row:
                # Merge with existing
                existing_ids = {
                    int(x) for x in zlib.decompress(row[0]).decode().split(',') if x
                }
                doc_ids = doc_ids.union(existing_ids)
            
            # Compress and store
            compressed = zlib.compress(
                ','.join(str(doc_id) for doc_id in sorted(doc_ids)).encode()
            )
            self.trigrams_db.execute(
                "INSERT OR REPLACE INTO trigrams (gram, doc_ids) VALUES (?, ?)",
                (gram, compressed)
            )
        
        self.trigrams_db.commit()
        return len(trigram_map)
    
    def _extract_trigrams(self, text: str) -> Set[str]:
        """Extract all trigrams from text."""
        trigrams = set()
        for i in range(len(text) - 2):
            trigrams.add(text[i:i+3])
        return trigrams
    
    def _read_blob(self, blob_sha: str) -> Optional[bytes]:
        """Read blob content from storage."""
        blob_file = self.index_path / "blobs" / blob_sha[:2] / blob_sha[2:]
        if blob_file.exists():
            return zlib.decompress(blob_file.read_bytes())
        return None
    
    def _should_skip(self, path: Path) -> bool:
        """Check if file should be skipped during indexing."""
        skip_dirs = {
            '.git', '__pycache__', 'node_modules', 'target',
            'build', 'dist', '.venv', 'venv', '.tox',
            '.mypy_cache', '.pytest_cache', 'coverage'
        }
        
        skip_extensions = {
            '.pyc', '.so', '.o', '.a', '.dylib', '.dll',
            '.exe', '.bin', '.pdf', '.png', '.jpg', '.gif',
            '.svg', '.ico', '.woff', '.woff2', '.ttf',
            '.zip', '.tar', '.gz', '.bz2', '.xz', '.mjs'
        }
        
        # Check if any parent is in skip_dirs
        for parent in path.parents:
            if parent.name in skip_dirs:
                return True
        
        # Check extension
        if path.suffix.lower() in skip_extensions:
            return True
        
        # Skip files starting with .
        if path.name.startswith('.'):
            return True
        
        # Skip Vite temporary build files (e.g., vite.config.ts.timestamp-*.mjs)
        if '.timestamp-' in path.name:
            return True
        
        # Skip large files (> 1MB)
        try:
            if path.stat().st_size > 1_000_000:
                return True
        except OSError:
            return True
        
        return False
    
    def search_code(
        self,
        query: str,
        repo: Optional[str] = None,
        max_results: int = 50
    ) -> List[SearchResult]:
        """
        Search for code using trigram index.
        
        Args:
            query: Search query (substring)
            repo: Optional repo name to restrict search
            max_results: Maximum number of results
        
        Returns:
            List of search results with context
        """
        query_lower = query.lower()
        query_trigrams = self._extract_trigrams(query_lower)
        logger.warning(
            "search_code: query=%s trigrams=%s",
            query,
            sorted(query_trigrams),
        )

        if not query_trigrams:
            logger.debug("search_code: no trigrams extracted for query %s", query)
            return []

        # Fetch document IDs for each trigram
        doc_id_sets = []
        for gram in query_trigrams:
            cursor = self.trigrams_db.execute(
                "SELECT doc_ids FROM trigrams WHERE gram = ?", (gram,)
            )
            row = cursor.fetchone()
            if row:
                doc_ids = {
                    int(x) for x in zlib.decompress(row[0]).decode().split(',') if x
                }
                logger.warning(
                    "search_code: trigram %s -> doc_ids=%s",
                    gram,
                    sorted(doc_ids),
                )
                doc_id_sets.append(doc_ids)
            else:
                logger.debug("search_code: trigram %s not found", gram)
                # Trigram not found, no results
                return []

        # Intersection of all doc_id sets
        candidate_doc_ids = set.intersection(*doc_id_sets)

        # Filter by repo if specified
        if repo:
            cursor = self.repos_db.execute(
                "SELECT id FROM repos WHERE name = ?", (repo,)
            )
            row = cursor.fetchone()
            if row:
                repo_id = row[0]
                stmt = (
                    "SELECT id FROM documents WHERE repo_id = ? AND id IN ({})"
                    .format(','.join('?' * len(candidate_doc_ids)))
                )
                cursor = self.repos_db.execute(stmt, (repo_id, *candidate_doc_ids))
                candidate_doc_ids = {row[0] for row in cursor.fetchall()}

        # Verify matches and extract context
        results = []
        for doc_id in candidate_doc_ids:
            if len(results) >= max_results:
                break

            doc = self._get_document(doc_id)
            if doc:
                content = self._read_blob(doc['blob_sha'])
                if content:
                    text = content.decode('utf-8', errors='replace')
                    # Find matching lines
                    for line_num, line in enumerate(text.splitlines(), start=1):
                        if query.lower() in line.lower():
                            results.append(SearchResult(
                                repo=doc['repo_name'],
                                path=doc['path'],
                                line=line_num,
                                text=line.strip(),
                                doc_id=f"{doc['repo_name']}::{doc['path']}"
                            ))
                            if len(results) >= max_results:
                                break

        return results
    
    def find_symbol(
        self,
        symbol_name: str,
        kind: Optional[str] = None,
        repo: Optional[str] = None
    ) -> List[Symbol]:
        """
        Find symbol definitions (IDE-like "Go to Definition").
        
        Args:
            symbol_name: Name of symbol to find
            kind: Optional symbol kind filter (function, class, etc.)
            repo: Optional repo name to restrict search
        
        Returns:
            List of symbol definitions
        """
        with self._lock:
            query = (
                "SELECT s.name, s.kind, s.line, s.signature, s.scope, "
                "d.path, r.name as repo_name "
                "FROM symbols s "
                "JOIN documents d ON s.doc_id = d.id "
                "JOIN repos r ON d.repo_id = r.id "
                "WHERE s.name = ?"
            )
            
            params = [symbol_name]
            
            if kind:
                query += " AND s.kind = ?"
                params.append(kind)
            
            if repo:
                query += " AND r.name = ?"
                params.append(repo)
            
            cursor = self.repos_db.execute(query, params)
            
            symbols = []
            for row in cursor.fetchall():
                symbols.append(Symbol(
                    name=row[0],
                    kind=row[1],
                    line=row[2],
                    signature=row[3],
                    scope=row[4],
                    file_path=f"{row[6]}::{row[5]}"  # repo::path
                ))
            
            return symbols
    
    def list_symbols(
        self,
        repo: str,
        file_path: Optional[str] = None,
        kind: Optional[str] = None
    ) -> List[Symbol]:
        """
        List symbols in a file or repository (IDE-like "Outline" view).
        
        Args:
            repo: Repository name
            file_path: Optional file path to restrict to
            kind: Optional symbol kind filter
        
        Returns:
            List of symbols
        """
        with self._lock:
            query = (
                "SELECT s.name, s.kind, s.line, s.signature, s.scope, d.path "
                "FROM symbols s "
                "JOIN documents d ON s.doc_id = d.id "
                "JOIN repos r ON d.repo_id = r.id "
                "WHERE r.name = ?"
            )
            
            params = [repo]
            
            if file_path:
                query += " AND d.path = ?"
                params.append(file_path)
            
            if kind:
                query += " AND s.kind = ?"
                params.append(kind)
            
            query += " ORDER BY d.path, s.line"
            
            cursor = self.repos_db.execute(query, params)
            
            symbols = []
            for row in cursor.fetchall():
                symbols.append(Symbol(
                    name=row[0],
                    kind=row[1],
                    line=row[2],
                    signature=row[3],
                    scope=row[4],
                    file_path=row[5]
                ))
            
            return symbols
    
    def _get_document(self, doc_id: int) -> Optional[dict[str, str]]:
        """Get document metadata."""
        cursor = self.repos_db.execute("""
            SELECT d.path, d.blob_sha, d.language, r.name as repo_name
            FROM documents d
            JOIN repos r ON d.repo_id = r.id
            WHERE d.id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                'path': row[0],
                'blob_sha': row[1],
                'language': row[2],
                'repo_name': row[3]
            }
        return None

    def _delete_symbols_and_embeddings_for_doc(self, doc_id: int) -> None:
        """Delete symbols and embeddings for a document id."""
        self.repos_db.execute("DELETE FROM symbols WHERE doc_id = ?", (doc_id,))
        self.repos_db.execute("DELETE FROM embeddings WHERE doc_id = ?", (doc_id,))

    def _remove_trigrams_for_doc_fast(self, doc_id: int, trigrams: Set[str]) -> None:
        """Fast path for removing doc_id from a list of trigrams."""
        for gram in trigrams:
            tri_cursor = self.trigrams_db.execute(
                "SELECT doc_ids FROM trigrams WHERE gram = ?",
                (gram,),
            )
            tri_row = tri_cursor.fetchone()
            if not tri_row:
                continue

            existing_ids = {
                int(x) for x in zlib.decompress(tri_row[0]).decode().split(",") if x
            }
            if doc_id not in existing_ids:
                logger.debug(
                    "remove_file: doc_id %s not in posting for gram %s",
                    doc_id,
                    gram,
                )
                continue

            existing_ids.remove(doc_id)
            if not existing_ids:
                logger.debug("remove_file: dropping trigram %s (no docs left)", gram)
                self.trigrams_db.execute("DELETE FROM trigrams WHERE gram = ?", (gram,))
            else:
                joined = ",".join(str(x) for x in sorted(existing_ids))
                compressed = zlib.compress(joined.encode())
                self.trigrams_db.execute(
                    "INSERT OR REPLACE INTO trigrams (gram, doc_ids) VALUES (?, ?)",
                    (gram, compressed),
                )
                logger.debug(
                    "remove_file: updated trigram %s -> %s",
                    gram,
                    sorted(existing_ids),
                )

    def _remove_trigrams_for_doc_scan(self, doc_id: int) -> None:
        """
        Slow fallback scanning removal: scan all trigrams and remove this doc_id
        where present.
        """
        tri_cursor = self.trigrams_db.execute("SELECT gram, doc_ids FROM trigrams")
        rows = tri_cursor.fetchall()
        for gram, blob in rows:
            existing_ids = {
                int(x) for x in zlib.decompress(blob).decode().split(",") if x
            }
            if doc_id not in existing_ids:
                continue

            existing_ids.remove(doc_id)
            if not existing_ids:
                self.trigrams_db.execute("DELETE FROM trigrams WHERE gram = ?", (gram,))
            else:
                joined = ",".join(str(x) for x in sorted(existing_ids))
                compressed = zlib.compress(joined.encode())
                self.trigrams_db.execute(
                    "INSERT OR REPLACE INTO trigrams (gram, doc_ids) VALUES (?, ?)",
                    (gram, compressed),
                )

    def _delete_blob_if_unreferenced(self, blob_sha: str, rel_path: str) -> None:
        """Delete blob file if no other document references it."""
        curs = self.repos_db.cursor()
        curs.execute("SELECT COUNT(*) FROM documents WHERE blob_sha = ?", (blob_sha,))
        ref_count = curs.fetchone()[0]
        if ref_count == 0:
            blob_file = self.index_path / "blobs" / blob_sha[:2] / blob_sha[2:]
            try:
                if blob_file.exists():
                    blob_file.unlink()
            except OSError:
                logger.debug(
                    "Failed to delete blob file %s for %s",
                    blob_file,
                    rel_path,
                )
    
    def _chunk_text(
        self,
        text: str,
        max_lines: int = 100,
        overlap: int = 10
    ) -> List[tuple[int, int, int, str]]:
        """
        Split text into overlapping chunks with line tracking.
        
        Args:
            text: Text to chunk
            max_lines: Maximum lines per chunk
            overlap: Number of overlapping lines between chunks
        
        Returns:
            List of (chunk_index, start_line, end_line, chunk_text) tuples
        """
        lines = text.splitlines()
        chunks = []
        i = 0
        chunk_idx = 0
        
        while i < len(lines):
            start = i
            end = min(i + max_lines, len(lines))
            if start >= end:
                break
            
            chunk_text = "\n".join(lines[start:end])
            chunks.append((chunk_idx, start + 1, end, chunk_text))  # 1-indexed lines
            chunk_idx += 1
            i += max_lines - overlap
        
        return chunks
    
    def build_vector_index(
        self,
        repo: str,
        embed_fn: Optional[EmbeddingFn] = None,
        model: Optional[str] = None,
        force: bool = False,
    ) -> dict[str, int]:
        """
        Build or refresh vector index for a repository.
        
        Args:
            repo: Repository name to index
            embed_fn: Embedding function (uses instance default if None)
            model: Model identifier (uses instance default if None)
            force: If True, rebuild existing embeddings
        
        Returns:
            Statistics about indexing operation
        """
        with self._lock:
            if embed_fn is None:
                embed_fn = self.embed_fn
            if embed_fn is None:
                raise RuntimeError("No embedding function configured for SigilIndex")
            
            model = model or self.embed_model
            
            stats = {
                "chunks_indexed": 0,
                "documents_processed": 0,
            }
            
            cur = self.repos_db.cursor()
            cur.execute("SELECT id FROM repos WHERE name = ?", (repo,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Repository {repo!r} not indexed yet")
            
            repo_id = row[0]
            
            # Optionally wipe old embeddings for this repo + model
            if force:
                cur.execute(
                    """
                    DELETE FROM embeddings
                    WHERE doc_id IN (SELECT id FROM documents WHERE repo_id = ?)
                    AND model = ?
                    """,
                    (repo_id, model),
                )
                self.repos_db.commit()
            
            cur.execute(
                "SELECT id, blob_sha FROM documents WHERE repo_id = ?",
                (repo_id,)
            )
            docs = cur.fetchall()
            
            for doc_id, blob_sha in docs:
                # Skip if already embedded (unless force)
                if not force:
                    cur2 = self.repos_db.execute(
                        "SELECT COUNT(*) FROM embeddings WHERE doc_id = ? "
                        "AND model = ?",
                        (doc_id, model),
                    )
                    if cur2.fetchone()[0] > 0:
                        continue
                
                content = self._read_blob(blob_sha)
                if not content:
                    continue
                
                text = content.decode("utf-8", errors="replace")
                chunks = self._chunk_text(text)
                if not chunks:
                    continue
                
                texts = [c[3] for c in chunks]
                vectors = embed_fn(texts)  # np.ndarray (N, dim)
                dim = int(vectors.shape[1])
                
                for (chunk_idx, start_line, end_line, _), vec in zip(chunks, vectors):
                    self.repos_db.execute("""
                        INSERT OR REPLACE INTO embeddings
                        (doc_id, chunk_index, start_line, end_line, model, dim, vector)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        doc_id,
                        chunk_idx,
                        start_line,
                        end_line,
                        model,
                        dim,
                        vec.astype("float32").tobytes(),
                    ))
                
                stats["documents_processed"] += 1
                stats["chunks_indexed"] += len(chunks)
            
            self.repos_db.commit()
            logger.info(
                "Built vector index for %s: %s documents, %s chunks",
                repo,
                stats['documents_processed'],
                stats['chunks_indexed'],
            )
            return stats
    
    def semantic_search(
        self,
        query: str,
        repo: str,
        k: int = 20,
        embed_fn: Optional[EmbeddingFn] = None,
        model: Optional[str] = None,
    ) -> List[dict[str, object]]:
        """
        Semantic code search using vector embeddings.
        
        Args:
            query: Natural language or code query
            repo: Repository name to search
            k: Number of top results to return
            embed_fn: Embedding function (uses instance default if None)
            model: Model identifier (uses instance default if None)
        
        Returns:
            List of search results with scores, sorted by relevance
        """
        with self._lock:
            if embed_fn is None:
                embed_fn = self.embed_fn
            if embed_fn is None:
                raise RuntimeError("No embedding function configured")
            
            model = model or self.embed_model
            
            # 1) embed query
            q_vec = embed_fn([query])[0].astype("float32")
            q_norm = np.linalg.norm(q_vec) or 1.0
            q_vec = q_vec / q_norm
            
            cur = self.repos_db.cursor()
            cur.execute("SELECT id FROM repos WHERE name = ?", (repo,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Repository {repo!r} not indexed yet")
            repo_id = row[0]
            
            # 2) fetch all embeddings for this repo + model
            cur.execute("""
                SELECT e.doc_id, e.chunk_index, e.start_line, e.end_line,
                       e.dim, e.vector, d.path
                FROM embeddings e
                JOIN documents d ON d.id = e.doc_id
                WHERE d.repo_id = ? AND e.model = ?
            """, (repo_id, model))
            
            rows = cur.fetchall()
            if not rows:
                return []
            
            # 3) compute cosine similarity in-memory
            vecs = []
            meta = []
            for doc_id, chunk_idx, start_line, end_line, dim, blob, path in rows:
                v = np.frombuffer(blob, dtype="float32")
                if v.shape[0] != dim:
                    continue
                vecs.append(v)
                meta.append((doc_id, chunk_idx, start_line, end_line, path))
            
            if not vecs:
                return []
            
            mat = np.stack(vecs, axis=0)
            norms = np.linalg.norm(mat, axis=1)
            norms[norms == 0] = 1.0
            mat = mat / norms[:, None]
            
            scores = mat @ q_vec
            top_idx = np.argsort(scores)[::-1][:k]
            
            # 4) map back to repo/path/lines
            results = []
            for idx in top_idx:
                score = float(scores[idx])
                doc_id, chunk_idx, start_line, end_line, path = meta[idx]
                doc = self._get_document(doc_id)
                if doc is None:
                    # Document was deleted but embedding still exists - skip it
                    continue
                results.append({
                    "repo": doc["repo_name"],
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "score": score,
                    "doc_id": f"{doc['repo_name']}::{path}",
                })
            
            return results
    
    def get_index_stats(self, repo: Optional[str] = None) -> dict[str, int | str]:
        """Get statistics about the index."""
        with self._lock:
            cursor = self.repos_db.cursor()
            
            if repo:
                cursor.execute("SELECT id FROM repos WHERE name = ?", (repo,))
                row = cursor.fetchone()
                if not row:
                    return {"error": "Repository not found"}
                repo_id = row[0]
                
                cursor.execute(
                    "SELECT COUNT(*) FROM documents WHERE repo_id = ?",
                    (repo_id,)
                )
                doc_count = cursor.fetchone()[0]
                
                cursor.execute(
                    "SELECT COUNT(*) FROM symbols WHERE doc_id IN "
                    "(SELECT id FROM documents WHERE repo_id = ?)",
                    (repo_id,)
                )
                symbol_count = cursor.fetchone()[0]
                
                cursor.execute(
                    "SELECT indexed_at FROM repos WHERE id = ?",
                    (repo_id,)
                )
                indexed_at = cursor.fetchone()[0]
                
                return {
                    "repo": repo,
                    "documents": doc_count,
                    "symbols": symbol_count,
                    "indexed_at": indexed_at
                }
            else:
                cursor.execute("SELECT COUNT(*) FROM repos")
                repo_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM documents")
                doc_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM symbols")
                symbol_count = cursor.fetchone()[0]
                
                # Query trigrams from the trigrams database
                tri_cursor = self.trigrams_db.cursor()
                tri_cursor.execute("SELECT COUNT(*) FROM trigrams")
                trigram_count = tri_cursor.fetchone()[0]
                
                return {
                    "repositories": repo_count,
                    "documents": doc_count,
                    "symbols": symbol_count,
                    "trigrams": trigram_count
                }

    def remove_file(
        self,
        repo_name: str,
        repo_path: Path,
        file_path: Path,
    ) -> bool:
        """
        Remove a single file from the index.

        This removes:
        - documents row
        - associated symbols
        - associated embeddings
        - this document's entries from trigram postings
        - blob content if no other documents reference it

        Returns:
            True if an indexed document was removed, False otherwise.
        """
        with self._lock:
            try:
                cursor = self.repos_db.cursor()

                # Resolve repo_id
                cursor.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
                row = cursor.fetchone()
                if not row:
                    return False
                repo_id = row[0]

                # Find document by repo + relative path
                rel_path = file_path.relative_to(repo_path).as_posix()
                cursor.execute(
                    "SELECT id, blob_sha FROM documents WHERE repo_id = ? AND path = ?",
                    (repo_id, rel_path),
                )
                row = cursor.fetchone()
                if not row:
                    return False

                doc_id, blob_sha = row
                logger.warning(
                    "remove_file: found doc_id=%s blob_sha=%s for %s",
                    doc_id,
                    blob_sha,
                    rel_path,
                )

                # Load content for trigram cleanup (optional but ideal).
                # If the blob is missing or unreadable, we still must ensure that
                # trigram postings do not retain this doc_id, even if it requires
                # a slower full-table scan as a fallback.
                content = self._read_blob(blob_sha)
                if content is not None:
                    text = content.decode("utf-8", errors="replace").lower()
                    trigrams = self._extract_trigrams(text)
                    logger.warning(
                        "remove_file: found %d trigrams for doc %s",
                        len(trigrams),
                        doc_id,
                    )
                else:
                    trigrams = None

                # Delete symbols and embeddings for this doc
                self._delete_symbols_and_embeddings_for_doc(doc_id)

                # Update trigram index to drop this doc_id
                if trigrams is not None:
                    # Fast path: we know which trigrams belonged to this document.
                    if trigrams:
                        self._remove_trigrams_for_doc_fast(doc_id, trigrams)
                else:
                    # Fallback scan to remove doc_id from any trigram postings found
                    self._remove_trigrams_for_doc_scan(doc_id)

                # Delete document row
                self.repos_db.execute(
                    "DELETE FROM documents WHERE id = ?",
                    (doc_id,),
                )

                # Optionally delete blob content if no other docs reference it
                self._delete_blob_if_unreferenced(blob_sha, rel_path)

                self.repos_db.commit()
                self.trigrams_db.commit()
                logger.warning(
                    "remove_file: committed deletion for doc_id %s "
                    "and updated trigrams",
                    doc_id,
                )

                logger.info("Removed %s from index (repo=%s)", rel_path, repo_name)
                return True
            except Exception as exc:
                logger.error("Error removing %s from index: %s", file_path, exc)
                return False
