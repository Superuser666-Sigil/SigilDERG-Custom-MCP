"""SQLite metadata store wrapper for Sigil MCP.

This is a thin encapsulation of the schema initialization and common helper
operations. It mirrors the schema currently created in `sigil_mcp.indexer` so
it can be swapped in incrementally.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MetadataStore:
    def __init__(self, db_path: Path, *, timeout: float = 60.0):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=timeout)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout=60000;")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create the minimal metadata schema used by the indexer."""
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS repos (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                path TEXT,
                indexed_at TEXT
            )
        """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                repo_id INTEGER,
                path TEXT,
                blob_sha TEXT,
                size INTEGER,
                language TEXT,
                FOREIGN KEY(repo_id) REFERENCES repos(id)
            )
        """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_doc_path
            ON documents(repo_id, path)
        """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_doc_blob_sha
            ON documents(blob_sha)
        """
        )

        cur.execute(
            """
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
        """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_symbol_name
            ON symbols(name)
        """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_symbol_kind
            ON symbols(kind)
        """
        )

        # Cached per-document trigram blob for quick removals/diffs
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS doc_trigrams (
                doc_id INTEGER PRIMARY KEY,
                trigrams BLOB
            )
        """
        )

        # Drop legacy embeddings table if present (safe no-op)
        try:
            cur.execute("DROP TABLE IF EXISTS embeddings")
        except Exception:
            logger.debug("Skipping removal of legacy embeddings table", exc_info=True)

        try:
            self.conn.commit()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            logger.debug("Error closing metadata database", exc_info=True)

    # Convenience helpers – callers may still use .conn directly during refactor
    def ensure_repo(self, name: str, path: str) -> int:
        """Ensure a repo row exists and return its id."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO repos (name, path, indexed_at) VALUES (?, ?, ?)",
            (name, path, datetime.now().isoformat()),
        )
        cur.execute("SELECT id FROM repos WHERE name = ?", (name,))
        row = cur.fetchone()
        return int(row[0])

    def get_repo_id(self, name: str) -> int | None:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM repos WHERE name = ?", (name,))
        row = cur.fetchone()
        return int(row[0]) if row else None

    def get_document(self, doc_id: int) -> dict[str, str] | None:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT d.path, d.blob_sha, d.language, r.name as repo_name
            FROM documents d
            JOIN repos r ON d.repo_id = r.id
            WHERE d.id = ?
        """,
            (doc_id,),
        )
        row = cur.fetchone()
        if row:
            return {"path": row[0], "blob_sha": row[1], "language": row[2], "repo_name": row[3]}
        return None

    def get_doc_trigrams_blob(self, doc_id: int) -> bytes | None:
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT trigrams FROM doc_trigrams WHERE doc_id = ?", (doc_id,))
            row = cur.fetchone()
            if not row or row[0] is None:
                return None
            return row[0]
        except Exception:
            logger.debug("Failed to get doc_trigrams for %s", doc_id, exc_info=True)
            return None

    def store_doc_trigrams_blob(self, doc_id: int, payload: bytes) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO doc_trigrams (doc_id, trigrams) VALUES (?, ?)",
                (doc_id, payload),
            )
            try:
                self.conn.commit()
            except Exception:
                pass
        except Exception:
            logger.debug("Failed to store doc_trigrams for %s", doc_id, exc_info=True)

    def delete_doc_trigram_record(self, doc_id: int) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM doc_trigrams WHERE doc_id = ?", (doc_id,))
            try:
                self.conn.commit()
            except Exception:
                pass
        except Exception:
            logger.debug("Failed to delete doc_trigrams for %s", doc_id, exc_info=True)



