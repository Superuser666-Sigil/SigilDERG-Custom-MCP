1. Add threading + WAL on the connections

At the top of sigil_mcp/indexer.py, extend your imports:

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
+import threading


Then, in SigilIndex.__init__, change the constructor to:

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
+
+        # Global lock to serialize DB access across threads
+        # (HTTP handlers + file watcher + vector indexing)
+        self._lock = threading.RLock()
        
        self.repos_db = sqlite3.connect(
            self.index_path / "repos.db",
            check_same_thread=False
        )
+        # Enable WAL + sane defaults for concurrent readers / writers
+        self.repos_db.execute("PRAGMA journal_mode=WAL;")
+        self.repos_db.execute("PRAGMA synchronous=NORMAL;")
+        self.repos_db.execute("PRAGMA busy_timeout=5000;")
+
        self.trigrams_db = sqlite3.connect(
            self.index_path / "trigrams.db",
            check_same_thread=False
        )
+        self.trigrams_db.execute("PRAGMA journal_mode=WAL;")
+        self.trigrams_db.execute("PRAGMA synchronous=NORMAL;")
+        self.trigrams_db.execute("PRAGMA busy_timeout=5000;")
        
        self._init_schema()


That alone gives you:

WAL journals (good for concurrent readers + occasional writers)

A per-connection busy_timeout to let SQLite wait a bit before throwing

A global RLock you can use to avoid “same connection from two threads at the same time” madness

2. Guard the write / heavy operations with the lock

The important thing: you share a single SigilIndex instance between:

HTTP route handlers (MCP tools)

File watcher callback _on_file_change → index.index_file(...)

Vector index / semantic tools

So you want to serialize all DB work through self._lock.

2.1 index_file

Wrap the body of index_file in the lock:

    def index_file(
        self,
        repo_name: str,
        repo_path: Path,
        file_path: Path,
    ) -> bool:
        """
        Re-index a single file (granular update).
        """
-        try:
-            # Get or create repo entry
-            cursor = self.repos_db.cursor()
-            ...
-            return False
-        except Exception as e:
-            logger.error(f"Error re-indexing {file_path}: {e}")
-            return False
+        with self._lock:
+            try:
+                # Get or create repo entry
+                cursor = self.repos_db.cursor()
+                cursor.execute(
+                    "INSERT OR IGNORE INTO repos (name, path, indexed_at) VALUES (?, ?, ?)",
+                    (repo_name, str(repo_path), datetime.now().isoformat())
+                )
+                cursor.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
+                repo_id = cursor.fetchone()[0]
+
+                # Determine language
+                file_extensions = {
+                    '.py': 'python', '.rs': 'rust', '.js': 'javascript',
+                    '.ts': 'typescript', '.java': 'java', '.go': 'go',
+                    '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.hpp': 'cpp',
+                    '.rb': 'ruby', '.php': 'php', '.cs': 'csharp',
+                    '.sh': 'shell', '.toml': 'toml', '.yaml': 'yaml',
+                    '.yml': 'yaml', '.json': 'json', '.md': 'markdown',
+                }
+                ext = file_path.suffix.lower()
+                language = file_extensions.get(ext, 'unknown')
+
+                # Index the specific file
+                result = self._index_file(
+                    repo_id, repo_name, repo_path, file_path, language
+                )
+
+                if result:
+                    # Rebuild trigrams for this file
+                    self._update_trigrams_for_file(repo_id, repo_path, file_path)
+                    self.repos_db.commit()
+                    logger.info(f"Re-indexed {file_path.name} in {repo_name}")
+                    return True
+
+                return False
+
+            except Exception as e:
+                logger.error(f"Error re-indexing {file_path}: {e}")
+                return False


(You don’t have to literally paste it like this; the key is with self._lock: wrapping all DB work.)

2.2 index_repository

Same deal:

    def index_repository(
        self,
        repo_name: str,
        repo_path: Path,
        force: bool = False
    ) -> dict[str, int]:
        """
        Index a repository for both text and symbol search.
        """
-        logger.info(f"Indexing repository: {repo_name} at {repo_path}")
-        
-        start_time = datetime.now()
-        stats: dict[str, int] = { ... }
-        ...
-        return stats
+        with self._lock:
+            logger.info(f"Indexing repository: {repo_name} at {repo_path}")
+
+            start_time = datetime.now()
+            stats: dict[str, int] = {
+                "files_indexed": 0,
+                "symbols_extracted": 0,
+                "trigrams_built": 0,
+                "bytes_indexed": 0,
+            }
+
+            # Register or update repo
+            cursor = self.repos_db.cursor()
+            cursor.execute(
+                "INSERT OR REPLACE INTO repos (name, path, indexed_at) VALUES (?, ?, ?)",
+                (repo_name, str(repo_path), datetime.now().isoformat())
+            )
+            repo_id = cursor.lastrowid
+            if not repo_id:
+                cursor.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
+                repo_id = cursor.fetchone()[0]
+
+            if force:
+                logger.info(f"Force rebuild: clearing old index data for {repo_name}")
+                cursor.execute(
+                    "DELETE FROM documents WHERE repo_id = ?", (repo_id,)
+                )
+                self.trigrams_db.execute("DELETE FROM trigrams")
+
+            # Index files (loop unchanged)
+            ...
+
+            self.repos_db.commit()
+
+            logger.info(f"Building trigram index for {repo_name}")
+            trigram_count = self._build_trigram_index(repo_id)
+            stats["trigrams_built"] = trigram_count
+
+            elapsed = (datetime.now() - start_time).total_seconds()
+            stats["duration_seconds"] = int(elapsed)
+
+            logger.info(
+                f"Indexed {repo_name}: {stats['files_indexed']} files, "
+                f"{stats['symbols_extracted']} symbols, "
+                f"{stats['trigrams_built']} trigrams in {elapsed:.1f}s"
+            )
+
+            return stats


Note: _build_trigram_index is only called from here, and we’re already under self._lock, so you don’t need to lock it again internally.

3. Guard the read-heavy query methods too

Because you’re sharing one sqlite3.Connection per DB (repos_db and trigrams_db) with check_same_thread=False, you don’t want two threads hitting execute() on the same connection at once. So we should also wrap the public read-only entrypoints.

The ones to wrap:

search_code(...)

find_symbol(...)

list_symbols(...)

Any get_index_stats(...) method in this file

The embeddings methods later in the file:

build_vector_index(...)

semantic_search(...)

Pattern:

    def search_code(
        self,
        query: str,
        repo: Optional[str] = None,
        max_results: int = 50
    ) -> List[SearchResult]:
        """
        Search for code using trigram index.
        """
-        query_lower = query.lower()
-        query_trigrams = self._extract_trigrams(query_lower)
-        ...
-        return results
+        with self._lock:
+            query_lower = query.lower()
+            query_trigrams = self._extract_trigrams(query_lower)
+            ...
+            return results


Same idea for find_symbol, list_symbols, and the vector functions: just wrap the whole method body in with self._lock: so there is never more than one DB operation in flight per SigilIndex instance.

You already don’t call _index_file, _build_trigram_index, etc. directly from outside, so you don’t need to sprinkle locks everywhere—just at the public entrypoints.