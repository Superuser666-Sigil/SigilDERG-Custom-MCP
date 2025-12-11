# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Hybrid code indexing for Sigil MCP Server.

Combines trigram-based text search with symbol extraction for IDE-like features.
Designed to work well with ChatGPT and other AI assistants via MCP.
"""

import hashlib
import zlib
import json
import subprocess
import os
import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Set, Callable, Sequence, Dict
import logging
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
import numpy as np
import threading

try:
    import lancedb  # type: ignore
    import pyarrow as pa  # type: ignore
    LANCEDB_AVAILABLE = True
except ImportError:
    lancedb = None  # type: ignore
    pa = None  # type: ignore
    LANCEDB_AVAILABLE = False

from .config import get_config
from .schema import get_code_chunk_model

logger = logging.getLogger(__name__)

# Type alias for embedding function: takes sequence of texts, returns (N, dim) array
EmbeddingFn = Callable[[Sequence[str]], np.ndarray]

USE_LANCEDB_STUB = os.getenv("SIGIL_MCP_LANCEDB_STUB", "").lower() == "1"


class _StubVectorType:
    def __init__(self, list_size: int):
        self.list_size = list_size


class _StubField:
    def __init__(self, dimension: int):
        self.type = _StubVectorType(dimension)


class _StubSchema:
    def __init__(self, dimension: int):
        self._dimension = dimension

    def field(self, name: str):
        return _StubField(self._dimension)


class _InMemoryArrowTable:
    """Minimal Arrow-like table wrapper for tests."""

    def __init__(self, rows: list[dict]):
        self._rows = list(rows)

    def to_pylist(self) -> list[dict]:
        return list(self._rows)


class _InMemoryQuery:
    """Simple query builder to mimic LanceDB search API in tests."""

    def __init__(self, rows: list[dict], query_vec: np.ndarray):
        self._rows = rows
        self._query_vec = query_vec.astype("float32") if query_vec is not None else None
        self._where = None
        self._limit: Optional[int] = None

    def where(self, expression: str):
        self._where = expression
        return self

    def limit(self, k: int):
        self._limit = k
        return self

    def _apply_filter(self) -> list[dict]:
        rows = list(self._rows)
        if not self._where:
            return rows

        def _extract(key: str) -> Optional[str]:
            match = re.search(rf"{key}\s*==\s*'([^']+)'", self._where or "")
            return match.group(1) if match else None

        repo_id = _extract("repo_id")
        file_path = _extract("file_path")
        doc_id = _extract("doc_id")

        filtered = []
        for row in rows:
            if repo_id is not None and str(row.get("repo_id")) != str(repo_id):
                continue
            if file_path is not None and str(row.get("file_path")) != str(file_path):
                continue
            if doc_id is not None and str(row.get("doc_id")) != str(doc_id):
                continue
            filtered.append(row)
        return filtered

    def to_list(self) -> list[dict]:
        rows = self._apply_filter()
        scored: list[tuple[float, dict]] = []
        for row in rows:
            vec = np.asarray(row.get("vector", []), dtype="float32")
            if self._query_vec is not None and vec.shape == self._query_vec.shape:
                score = float(np.dot(vec, self._query_vec))
            else:
                score = 0.0
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [row for _, row in scored]
        if self._limit is not None:
            results = results[: self._limit]
        return results


class InMemoryLanceTable:
    """Lightweight in-memory stand-in for LanceDB tables when real LanceDB is unavailable."""

    def __init__(self, name: str, dimension: int):
        self.name = name
        self.dimension = dimension
        self.rows: list[dict] = []
        self.schema = _StubSchema(dimension)

    def _normalize_record(self, record: object) -> dict:
        if hasattr(record, "model_dump"):
            data = record.model_dump()
        elif hasattr(record, "dict"):
            data = record.dict()  # type: ignore[attr-defined]
        else:
            data = dict(record)  # type: ignore[arg-type]

        vec = data.get("vector")
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        data["vector"] = vec
        return data

    def count_rows(self) -> int:
        return len(self.rows)

    def to_arrow(self):
        return _InMemoryArrowTable(self.rows)

    def to_list(self, limit: Optional[int] = None) -> list[dict]:
        if limit is None:
            return list(self.rows)
        return list(self.rows[:limit])

    def head(self, limit: int):
        return _InMemoryArrowTable(self.rows[:limit])

    def delete(self, clause: str):
        repo_match = re.search(r"repo_id\s*==\s*'([^']+)'", clause or "")
        file_match = re.search(r"file_path\s*==\s*'([^']+)'", clause or "")
        doc_match = re.search(r"doc_id\s*==\s*'([^']+)'", clause or "")

        def _keep(row: dict) -> bool:
            if repo_match and str(row.get("repo_id")) != repo_match.group(1):
                return True
            if file_match and str(row.get("file_path")) != file_match.group(1):
                return True
            if doc_match and str(row.get("doc_id")) != doc_match.group(1):
                return True
            # If any clause matched and row satisfied, drop it
            if repo_match or file_match or doc_match:
                return False
            return True

        self.rows = [r for r in self.rows if _keep(r)]

    def add(self, records: list[object]):
        for rec in records:
            self.rows.append(self._normalize_record(rec))

    def update(self, where: str, values: dict):
        doc_match = re.search(r"doc_id\\s*==\\s*'([^']+)'", where or "")
        target_doc = doc_match.group(1) if doc_match else None
        for row in self.rows:
            if target_doc is None or str(row.get("doc_id")) == str(target_doc):
                row.update(values)

    def search(self, query_vec: np.ndarray) -> _InMemoryQuery:
        return _InMemoryQuery(self.rows, query_vec)


class InMemoryLanceDB:
    """Minimal LanceDB-like interface storing vectors in memory for tests."""

    def __init__(self, default_dimension: int = 768):
        self.default_dimension = default_dimension
        self.tables: dict[str, InMemoryLanceTable] = {}

    def table_names(self) -> list[str]:
        return list(self.tables.keys())

    def open_table(self, name: str) -> InMemoryLanceTable:
        return self.tables[name]

    def create_table(self, name: str, schema=None, mode: Optional[str] = None, **_kwargs) -> InMemoryLanceTable:
        if mode == "overwrite" or name not in self.tables:
            dimension = self.default_dimension
            if schema is not None and hasattr(schema, "__name__"):
                try:
                    vector_field = getattr(schema, "model_fields", {}).get("vector")
                    if vector_field and getattr(vector_field.annotation, "size", None):
                        dimension = int(vector_field.annotation.size)
                except Exception:
                    dimension = self.default_dimension
            self.tables[name] = InMemoryLanceTable(name, dimension)
        return self.tables[name]


_STUB_LANCEDB_REGISTRY: Dict[str, InMemoryLanceDB] = {}

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
        self._embedding_init_failed = False
        self.use_lancedb_stub = USE_LANCEDB_STUB

        config = get_config()
        self.embedding_dimension = config.embeddings_dimension
        self.embedding_provider = config.embeddings_provider
        self.allow_vector_schema_overwrite = config.index_allow_vector_schema_overwrite
        if config.index_path == self.index_path:
            self.lance_db_path = config.lance_dir
        else:
            self.lance_db_path = self.index_path / "lancedb"
        self.lance_db = None
        self.lance = None
        self.vector_table_name = "code_vectors"
        self.vectors = None
        self.lancedb_available = LANCEDB_AVAILABLE or self.use_lancedb_stub
        self._vector_index_stale = False

        # If embeddings are enabled via config but no embed_fn was provided,
        # try to initialize one automatically so semantic search works out-of-the-box.
        if self.embed_fn is None and config.embeddings_enabled:
            self._auto_initialize_embed_fn_from_config(config)

        self._embeddings_requested = config.embeddings_enabled or self.embed_fn is not None
        self._embeddings_active = self._embeddings_requested and self.lancedb_available
        if self.embed_fn is not None and not config.embeddings_enabled:
            logger.warning(
                "Embedding function provided but embeddings.enabled is False; "
                "LanceDB will still be initialized for vector storage."
            )
        if self._embeddings_requested and not self.lancedb_available:
            logger.warning(
                "Embeddings requested but LanceDB/pyarrow are not installed. "
                "Install with `pip install .[lancedb]` or disable embeddings to use "
                "trigram-only search."
            )

        if self._embeddings_active:
            self._ensure_lance_dir_permissions()
            try:
                if self.use_lancedb_stub:
                    stub_key = str(self.lance_db_path)
                    if stub_key in _STUB_LANCEDB_REGISTRY:
                        self.lance_db = _STUB_LANCEDB_REGISTRY[stub_key]
                    else:
                        self.lance_db = InMemoryLanceDB(self.embedding_dimension)
                        _STUB_LANCEDB_REGISTRY[stub_key] = self.lance_db
                else:
                    self.lance_db = lancedb.connect(str(self.lance_db_path)) if lancedb else None
            except Exception as exc:
                logger.error(
                    "Failed to initialize LanceDB at %s: %s. Falling back to trigram-only search.",
                    self.lance_db_path,
                    exc,
                )
                self.lance_db = None
                self._embeddings_active = False
            self.lance = self.lance_db
            self._code_chunk_model = get_code_chunk_model(self.embedding_dimension)
            if self.lance_db is not None:
                table_names = set(self.lance_db.table_names())
                target_table = self.vector_table_name
                if target_table in table_names:
                    self.vectors = self.lance_db.open_table(target_table)
                elif "code_chunks" in table_names:
                    target_table = "code_chunks"
                    self.vector_table_name = target_table
                    self.vectors = self.lance_db.open_table(target_table)
                else:
                    self.vectors = self.lance_db.create_table(
                        target_table, schema=self._code_chunk_model
                    )
            self._sync_embedding_dimension_from_lance()
            self._log_embedding_startup()
            self._warn_on_vector_schema_mismatch()
        else:
            self._code_chunk_model = None
            self.vectors = None

        self._vector_index_enabled = self._embeddings_active and self.vectors is not None
        self._log_vector_index_status()

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
        self._check_vector_repo_alignment()

    def _auto_initialize_embed_fn_from_config(self, config) -> None:
        """Initialize an embedding function from config when none is provided."""

        provider = config.embeddings_provider
        model = config.embeddings_model

        if not provider or not model:
            logger.info(
                "Embeddings enabled but provider/model not configured; "
                "semantic search will remain disabled until configured."
            )
            return

        # Ensure we reflect the configured model in logs even if initialization fails
        if self.embed_model in {"none", "local"}:
            self.embed_model = f"{provider}:{model}"

        try:
            # Import lazily so we only pull heavy deps when needed
            from .embeddings import create_embedding_provider

            kwargs = dict(config.embeddings_kwargs)
            if config.embeddings_cache_dir:
                kwargs["cache_dir"] = config.embeddings_cache_dir
            if provider == "openai" and config.embeddings_api_key:
                kwargs["api_key"] = config.embeddings_api_key

            provider_impl = create_embedding_provider(
                provider=provider,
                model=model,
                dimension=self.embedding_dimension,
                **kwargs,
            )

            def _embed(texts: Sequence[str]) -> np.ndarray:
                embeddings_list = provider_impl.embed_documents(list(texts))
                return np.asarray(embeddings_list, dtype="float32")

            self.embed_fn = _embed
            self.embed_model = f"{provider}:{model}"
            logger.info(
                "Initialized embedding provider from config: provider=%s model=%s dim=%s",
                provider,
                model,
                self.embedding_dimension,
            )
        except Exception:
            self._embedding_init_failed = True
            logger.exception(
                "Failed to initialize embedding provider (provider=%s, model=%s); "
                "semantic search will be unavailable until an embed_fn is provided.",
                provider,
                model,
            )

    def _ensure_lance_dir_permissions(self) -> None:
        """Create the LanceDB path and align its permissions to the index dir."""

        created = False
        if not self.lance_db_path.exists():
            self.lance_db_path.mkdir(parents=True, exist_ok=True)
            created = True
        else:
            self.lance_db_path.mkdir(parents=True, exist_ok=True)

        if created:
            try:
                mode = self.index_path.stat().st_mode & 0o777
                os.chmod(self.lance_db_path, mode)
            except Exception:
                logger.exception(
                    "Failed to apply index directory permissions to LanceDB path %s",
                    self.lance_db_path,
                )

    def _log_embedding_startup(self) -> None:
        """Log embedding provider configuration for visibility at startup."""

        if self.vectors is None:
            logger.info(
                "Vector index inactive at startup: embeddings_requested=%s, "
                "lancedb_available=%s",
                self._embeddings_requested,
                self.lancedb_available,
            )
            return

        logger.info(
            "Embeddings enabled: provider=%s model=%s dim=%s lance_path=%s",
            self.embedding_provider,
            self.embed_model,
            self.embedding_dimension,
            self.lance_db_path,
        )

        if self.embed_fn is None:
            logger.warning(
                "LanceDB initialized but no embedding function configured; "
                "vector indexing calls will fail until embeddings are set."
            )

    def _sync_embedding_dimension_from_lance(self) -> None:
        """Align configured embedding dimension with existing LanceDB schema."""

        if self.vectors is None or pa is None:
            return

        try:
            vector_field = self.vectors.schema.field("vector")
            vector_type = vector_field.type
            actual_dim = None
            if isinstance(vector_type, pa.FixedSizeListType):
                actual_dim = vector_type.list_size
        except Exception:
            logger.debug("Could not inspect LanceDB vector schema", exc_info=True)
            return

        if actual_dim and actual_dim != self.embedding_dimension:
            logger.warning(
                "Configured embedding dimension %s does not match LanceDB table "
                "dimension %s, using table dimension.",
                self.embedding_dimension,
                actual_dim,
            )
            self.embedding_dimension = actual_dim
            self._code_chunk_model = get_code_chunk_model(actual_dim)

    def _warn_on_vector_schema_mismatch(self) -> None:
        """Warn if the LanceDB schema does not match configured dimensions."""

        if self.vectors is None or pa is None:
            return

        try:
            vector_field = self.vectors.schema.field("vector")
            vector_type = vector_field.type
            actual_dim = None
            if isinstance(vector_type, pa.FixedSizeListType):
                actual_dim = vector_type.list_size
        except Exception:
            logger.debug("Could not inspect LanceDB vector schema", exc_info=True)
            return

        if actual_dim and actual_dim != self.embedding_dimension:
            logger.warning(
                "Configured embedding dimension %s does not match LanceDB table "
                "dimension %s. Consider rebuilding embeddings to avoid mismatches.",
                self.embedding_dimension,
                actual_dim,
            )
    
    def _log_vector_index_status(self, context: str = "startup") -> None:
        """Log current vector index availability and size."""

        if self.vectors is None:
            logger.info(
                "Vector index unavailable (%s); trigram search will be used.",
                context,
            )
            return

        try:
            row_count = int(self.vectors.count_rows())
        except Exception:
            logger.debug(
                "Failed to count vector rows during %s status check",
                context,
                exc_info=True,
            )
            row_count = -1

        logger.info(
            "Vector index ready (%s): %s indexed chunks at %s",
            context,
            row_count,
            self.lance_db_path,
        )

    def _sample_vector_repo_ids(self, limit: int = 200) -> Set[str]:
        """Read a small sample of repo_ids from LanceDB for sanity checks."""

        if self.vectors is None:
            return set()

        try:
            if hasattr(self.vectors, "to_list"):
                # Older LanceDB versions exposed to_list on LanceTable
                rows = self.vectors.to_list(limit=limit)
            else:
                # Newer versions expose head() on LanceTable and to_list() on queries
                table = self.vectors.head(limit)
                rows = table.to_pylist() if table is not None else []
        except Exception:
            logger.exception("Failed to sample repo_ids from vector table")
            return set()

        return {
            str(r.get("repo_id"))
            for r in rows
            if r.get("repo_id") is not None
        }

    def _check_vector_repo_alignment(self) -> None:
        """
        Detect repo-id drift between LanceDB and repos.db.

        If repo IDs in the vector table do not overlap any known repos,
        semantic search is marked stale to avoid silently returning no results.
        """

        if not self._vector_index_enabled or self.vectors is None:
            return

        repo_ids = {
            str(row[0])
            for row in self.repos_db.execute("SELECT id FROM repos")
        }
        sample_repo_ids = self._sample_vector_repo_ids()

        if not repo_ids or not sample_repo_ids:
            return

        if sample_repo_ids.isdisjoint(repo_ids):
            self._vector_index_stale = True
            logger.warning(
                "Vector index appears stale: repo_ids in LanceDB=%s do not "
                "match repos table ids=%s. Semantic search will be disabled "
                "until the vector index is rebuilt (remove LanceDB dir or "
                "trigger rebuild_embeddings).",
                sorted(sample_repo_ids),
                sorted(repo_ids),
            )
    
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
                blob_sha TEXT,
                size INTEGER,
                language TEXT,
                FOREIGN KEY(repo_id) REFERENCES repos(id)
            )
        """)
        
        self.repos_db.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_path 
            ON documents(repo_id, path)
        """)

        # Ensure blob_sha is indexed (but not unique) for reuse/lookups
        self.repos_db.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_blob_sha
            ON documents(blob_sha)
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
        
        # Remove legacy embeddings table now that vectors are stored in LanceDB
        try:
            self.repos_db.execute("DROP TABLE IF EXISTS embeddings")
        except Exception:
            logger.debug("Skipping removal of legacy embeddings table", exc_info=True)

        # Keep 'repos' table
        # Replace 'file_content_fts' creation with:
        self.trigrams_db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS file_content_fts 
            USING fts5(
                path UNINDEXED, 
                repo_id UNINDEXED, 
                content, 
                tokenize='trigram'
            )
        """)

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
                    if self.vectors is not None and self.embed_fn is not None:
                        chunks = self._chunk_text(result.get("text", ""))
                        if chunks:
                            embeddings = self.embed_fn([c[3] for c in chunks])
                            self._index_file_vectors(
                                repo_id,
                                result.get("doc_id", 0),
                                result.get("rel_path", file_path.as_posix()),
                                chunks,
                                embeddings,
                            )

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
                """
                INSERT INTO repos (name, path, indexed_at)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    path=excluded.path,
                    indexed_at=excluded.indexed_at
                """,
                (repo_name, str(repo_path), datetime.now().isoformat()),
            )
            # Preserve stable repo_id (INSERT ... ON CONFLICT does not replace row)
            cursor.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
            repo_id = cursor.fetchone()[0]
            
            # Clear old repo-scoped data if forcing rebuild
            if force:
                cursor.execute(
                    "SELECT id FROM documents WHERE repo_id = ?",
                    (repo_id,),
                )
                old_doc_ids = {row[0] for row in cursor.fetchall()}
                logger.info(
                    "Force rebuild: clearing old index data for %s (%s docs)",
                    repo_name,
                    len(old_doc_ids),
                )

                if old_doc_ids:
                    placeholders = ",".join("?" for _ in old_doc_ids)
                    cursor.execute(
                        f"DELETE FROM symbols WHERE doc_id IN ({placeholders})",
                        tuple(old_doc_ids),
                    )
                cursor.execute(
                    "DELETE FROM documents WHERE repo_id = ?", (repo_id,)
                )
                # Remove only this repo's postings from trigram index
                self._remove_trigrams_for_doc_ids(old_doc_ids)
            
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
    ) -> Optional[dict[str, object]]:
        """Index a single file."""
        try:
            content = file_path.read_bytes()
            text = content.decode("utf-8", errors="replace")
            blob_sha = hashlib.sha256(content).hexdigest()
            rel_path = file_path.relative_to(repo_root).as_posix()
            cursor = self.repos_db.cursor()

            # Check if this repo/path is already indexed
            cursor.execute(
                "SELECT id, blob_sha FROM documents WHERE repo_id = ? AND path = ?",
                (repo_id, rel_path),
            )
            existing = cursor.fetchone()

            if existing:
                existing_doc_id, existing_blob = existing
                if existing_blob == blob_sha:
                    # Already indexed with same content; refresh metadata and skip work
                    cursor.execute(
                        "UPDATE documents SET language = ?, size = ? WHERE id = ?",
                        (language, len(content), existing_doc_id),
                    )
                    self._update_vector_metadata_for_doc(
                        doc_id=existing_doc_id,
                        repo_id=repo_id,
                        rel_path=rel_path,
                    )
                    return None

                # Path is the same but content changed: clean up old symbols/vectors/doc
                self._delete_symbols_and_embeddings_for_doc(
                    existing_doc_id, repo_id, rel_path
                )
                cursor.execute("DELETE FROM documents WHERE id = ?", (existing_doc_id,))

            # Store document metadata (repo/path scoped; blob_sha can be reused across repos)
            # Remove any additional stale rows for this repo/path just in case
            try:
                cursor.execute(
                    "SELECT id FROM documents WHERE repo_id = ? AND path = ?",
                    (repo_id, rel_path)
                )
                rows = cursor.fetchall()
                for (old_doc_id,) in rows:
                    self._delete_symbols_and_embeddings_for_doc(
                        old_doc_id, repo_id, rel_path
                    )
                    cursor.execute("DELETE FROM documents WHERE id = ?", (old_doc_id,))
            except Exception:
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
                "bytes": int(len(content)),
                "doc_id": doc_id,
                "rel_path": rel_path,
                "text": text,
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

    def _remove_trigrams_for_doc_ids(self, doc_ids: Set[int]) -> None:
        """
        Remove trigram postings that reference the provided document IDs.

        This lets us rebuild a single repo without wiping trigram data for
        every other repo.
        """
        if not doc_ids:
            return

        cursor = self.trigrams_db.cursor()
        updates: list[tuple[bytes, str]] = []
        deletes: list[tuple[str]] = []
        processed = 0

        for gram, compressed in cursor.execute("SELECT gram, doc_ids FROM trigrams"):
            try:
                existing_ids = {
                    int(x)
                    for x in zlib.decompress(compressed).decode().split(",")
                    if x
                }
            except Exception:
                logger.exception("Failed to decode trigram row for %s", gram)
                continue

            if not existing_ids.intersection(doc_ids):
                continue

            remaining = existing_ids.difference(doc_ids)
            if remaining:
                new_blob = zlib.compress(
                    ",".join(str(x) for x in sorted(remaining)).encode()
                )
                updates.append((new_blob, gram))
            else:
                deletes.append((gram,))

            processed += 1
            if processed % 500 == 0:
                if updates:
                    cursor.executemany(
                        "UPDATE trigrams SET doc_ids = ? WHERE gram = ?",
                        updates,
                    )
                    updates.clear()
                if deletes:
                    cursor.executemany(
                        "DELETE FROM trigrams WHERE gram = ?", deletes
                    )
                    deletes.clear()

        if updates:
            cursor.executemany(
                "UPDATE trigrams SET doc_ids = ? WHERE gram = ?", updates
            )
        if deletes:
            cursor.executemany("DELETE FROM trigrams WHERE gram = ?", deletes)

        self.trigrams_db.commit()
    
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
        skip_dirs.update({"htmlcov", "coverage_html", ".coverage", "site-packages"})
        
        skip_extensions = {
            '.pyc', '.so', '.o', '.a', '.dylib', '.dll',
            '.exe', '.bin', '.pdf', '.png', '.jpg', '.gif',
            '.svg', '.ico', '.woff', '.woff2', '.ttf',
            '.zip', '.tar', '.gz', '.bz2', '.xz', '.mjs'
        }
        skip_extensions.update({'.html', '.htm'})
        
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
        # The database connection is thread-safe for reads in WAL mode.
        start = perf_counter()
        query_lower = query.lower()
        query_trigrams = self._extract_trigrams(query_lower)
        logger.debug("search_code: query=%s trigrams=%s", query, sorted(query_trigrams))

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
                logger.debug("search_code: trigram %s -> doc_ids=%s", gram, sorted(doc_ids))
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

        duration = perf_counter() - start
        logger.info(
            "search_code completed repo=%s query=%r results=%s duration=%.3fs",
            repo or "*",
            query,
            len(results),
            duration,
        )

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

    def _delete_symbols_and_embeddings_for_doc(
        self,
        doc_id: int,
        repo_id: int | None = None,
        rel_path: str | None = None,
    ) -> None:
        """Delete symbols and embeddings for a document id.

        When LanceDB is enabled, this also deletes vector rows keyed by doc_id
        and (repo_id, file_path) to ensure stale chunks are removed even if
        doc_id reuse or path changes occur.
        """
        self.repos_db.execute("DELETE FROM symbols WHERE doc_id = ?", (doc_id,))
        try:
            self.repos_db.execute("DELETE FROM embeddings WHERE doc_id = ?", (doc_id,))
        except Exception:
            # Embeddings table may not exist in newer versions; ignore errors
            logger.debug("No embeddings table found when deleting doc %s", doc_id)

        if self.vectors is not None:
            delete_clauses = [f"doc_id == '{doc_id}'"]
            if repo_id is not None and rel_path is not None:
                delete_clauses.append(
                    f"repo_id == '{repo_id}' AND file_path == '{rel_path}'"
                )

            for clause in delete_clauses:
                try:
                    self.vectors.delete(clause)
                except Exception:
                    logger.exception(
                        "Failed to delete vector rows for doc %s using clause %s",
                        doc_id,
                        clause,
                    )

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

    def _update_vector_metadata_for_doc(
        self,
        doc_id: int,
        repo_id: int,
        rel_path: str,
    ) -> None:
        """Keep vector rows in sync when a blob moves to a different repo/path."""

        if self.vectors is None:
            return

        try:
            self.vectors.update(
                where=f"doc_id == '{doc_id}'",
                values={
                    "repo_id": str(repo_id),
                    "file_path": rel_path,
                },
            )
        except Exception:
            logger.exception(
                "Failed to update vector metadata for doc %s (repo %s, path %s)",
                doc_id,
                repo_id,
                rel_path,
            )

    def _index_file_vectors(
        self,
        repo_id: int,
        doc_id: int,
        rel_path: str,
        chunks: Sequence[tuple[int, int, int, str]],
        embeddings: np.ndarray,
    ) -> None:
        """Replace vector rows for a file with fresh embeddings."""

        if self.vectors is None:
            return

        if embeddings.shape[0] != len(chunks):
            logger.warning(
                "Embedding/chunk count mismatch for %s: %s embeddings vs %s chunks",
                rel_path,
                embeddings.shape[0],
                len(chunks),
            )
            return

        if embeddings.shape[1] != self.embedding_dimension:
            new_dim = int(embeddings.shape[1])
            logger.error(
                "Embedding dimension mismatch for %s: incoming=%s configured=%s "
                "(will %s the vector table)",
                rel_path,
                new_dim,
                self.embedding_dimension,
                "overwrite" if self.allow_vector_schema_overwrite else "NOT overwrite",
            )

            if not self.allow_vector_schema_overwrite:
                # Mark stale so semantic_search will refuse to run until rebuilt.
                self._vector_index_stale = True
                return

            self.embedding_dimension = new_dim
            self._code_chunk_model = get_code_chunk_model(new_dim)
            try:
                self.vectors = self.lance_db.create_table(
                    self.vector_table_name,
                    schema=self._code_chunk_model,
                    mode="overwrite",
                )
            except Exception:
                logger.exception(
                    "Failed to recreate vector table with dimension %s for %s",
                    new_dim,
                    rel_path,
                )
                return

        timestamp = datetime.now()

        try:
            self.vectors.delete(
                f"repo_id == '{repo_id}' AND file_path == '{rel_path}'"
            )
        except Exception:
            logger.exception("Failed to delete existing vectors for %s", rel_path)

        records = []
        code_chunk_model = self._code_chunk_model or get_code_chunk_model(
            self.embedding_dimension
        )
        for (chunk_idx, start_line, end_line, chunk_text), vector in zip(chunks, embeddings):
            records.append(code_chunk_model(
                vector=np.asarray(vector, dtype="float32"),
                doc_id=str(doc_id),
                repo_id=str(repo_id),
                file_path=rel_path,
                chunk_index=int(chunk_idx),
                start_line=int(start_line),
                end_line=int(end_line),
                content=chunk_text,
                last_updated=timestamp,
            ))

        if records:
            try:
                self.vectors.add(records)
            except Exception:
                logger.exception("Failed to upsert vectors for %s", rel_path)
    
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
            if not self._embeddings_active or not self.lancedb_available:
                logger.info(
                    "Embeddings disabled or LanceDB unavailable; skipping vector index build for %s",
                    repo,
                )
                return {"chunks_indexed": 0, "documents_processed": 0}

            if embed_fn is None:
                embed_fn = self.embed_fn
            if embed_fn is None:
                logger.warning(
                    "No embedding function configured for SigilIndex; skipping vector build for %s",
                    repo,
                )
                return {"chunks_indexed": 0, "documents_processed": 0}
            
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

            if self.vectors is None:
                logger.info("Vector index not initialized; skipping build for %s", repo)
                return stats

            if force and self.vectors is not None:
                try:
                    self.vectors.delete(f"repo_id == '{repo_id}'")
                except Exception:
                    logger.exception(
                        "Failed to clear existing vectors for repo %s", repo
                    )
            
            cur.execute(
                "SELECT id, blob_sha, path FROM documents WHERE repo_id = ?",
                (repo_id,)
            )
            docs = cur.fetchall()

            for doc_id, blob_sha, rel_path in docs:
                content = self._read_blob(blob_sha)
                if not content:
                    continue

                text = content.decode("utf-8", errors="replace")
                chunks = self._chunk_text(text)
                if not chunks:
                    continue

                texts = [c[3] for c in chunks]
                vectors = embed_fn(texts)  # np.ndarray (N, dim)

                self._index_file_vectors(repo_id, doc_id, rel_path, chunks, vectors)

                stats["documents_processed"] += 1
                stats["chunks_indexed"] += len(chunks)

            logger.info(
                "Built vector index for %s using model %s: %s documents, %s chunks",
                repo,
                model,
                stats['documents_processed'],
                stats['chunks_indexed'],
            )
            self._log_vector_index_status(context=f"rebuild:{repo}")
            return stats
    
    def semantic_search(
        self,
        query: str,
        repo: Optional[str] = None,
        k: int = 20,
        embed_fn: Optional[EmbeddingFn] = None,
        model: Optional[str] = None,
    ) -> List[dict[str, object]]:
        """
        Semantic code search using vector embeddings.
        
        Args:
            query: Natural language or code query
            repo: Repository name to search (optional; searches all if omitted)
            k: Number of top results to return
            embed_fn: Embedding function (uses instance default if None)
            model: Model identifier (uses instance default if None)
        
        Returns:
            List of search results with scores, sorted by relevance
        """
        with self._lock:
            if embed_fn is None:
                embed_fn = self.embed_fn
            if embed_fn is None or self.vectors is None:
                logger.info(
                    "Semantic search requested but embeddings are unavailable; "
                    "returning no results."
                )
                return []

            if self._vector_index_stale:
                msg = (
                    "Vector index repo IDs are stale; rebuild embeddings or "
                    "clear LanceDB to realign (semantic search disabled)."
                )
                logger.warning(msg)
                raise RuntimeError(msg)

            model = model or self.embed_model

            # 1) embed query
            q_vec = embed_fn([query])[0].astype("float32")
            q_norm = np.linalg.norm(q_vec) or 1.0
            q_vec = q_vec / q_norm

            repo_id: Optional[str] = None
            repo_lookup: dict[str, str] = {}
            query_builder = self.vectors.search(q_vec.astype("float32"))

            if repo:
                cur = self.repos_db.cursor()
                cur.execute("SELECT id FROM repos WHERE name = ?", (repo,))
                row = cur.fetchone()
                if not row:
                    raise ValueError(f"Repository {repo!r} not indexed yet")
                repo_id = str(row[0])
                query_builder = query_builder.where(f"repo_id == '{repo_id}'")
                repo_lookup[repo_id] = repo

            query_results = query_builder.limit(k).to_list()

            results = []
            for row in query_results:
                try:
                    doc_id = int(row.get("doc_id", 0))
                except (TypeError, ValueError):
                    continue

                repo_name: Optional[str] = None
                if repo_id is not None:
                    repo_name = repo_lookup.get(repo_id)
                else:
                    doc = self._get_document(doc_id)
                    if doc is not None:
                        repo_name = doc.get("repo_name")
                        if doc.get("repo_name"):
                            repo_lookup[str(row.get("repo_id", ""))] = doc["repo_name"]
                if repo_name is None:
                    continue

                distance = float(row.get("_distance", 0.0))
                score = 1.0 / (1.0 + distance)

                results.append({
                    "repo": repo_name,
                    "path": row.get("file_path", ""),
                    "chunk_index": int(row.get("chunk_index", -1)),
                    "start_line": int(row.get("start_line", 0)),
                    "end_line": int(row.get("end_line", 0)),
                    "content": row.get("content", ""),
                    "score": score,
                    "doc_id": f"{repo_name}::{row.get('file_path', '')}",
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

                # Delete symbols and embeddings for this doc and clear vectors
                self._delete_symbols_and_embeddings_for_doc(
                    doc_id, repo_id, rel_path
                )

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
