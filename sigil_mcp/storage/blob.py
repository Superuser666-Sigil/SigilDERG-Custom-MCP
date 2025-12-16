"""Blob storage helpers for compressed file blobs used by the indexer.

This extracts the tiny bits of filesystem logic out of `indexer.py` so the
indexer can act as a coordinator while the storage module owns the on-disk
layout and helpers for reading/writing/deleting blobs.
"""
from __future__ import annotations

import logging
import zlib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BlobStore:
    def __init__(self, base_path: Path, repos_conn: Any | None = None):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.repos_conn = repos_conn

    def write_blob(self, blob_sha: str, content: bytes) -> None:
        blob_dir = self.base_path / blob_sha[:2]
        try:
            blob_dir.mkdir(parents=True, exist_ok=True)
            blob_file = blob_dir / blob_sha[2:]
            if not blob_file.exists():
                blob_file.write_bytes(zlib.compress(content))
        except Exception:
            logger.debug("Failed to write blob %s", blob_sha, exc_info=True)

    def read_blob(self, blob_sha: str) -> bytes | None:
        blob_file = self.base_path / blob_sha[:2] / blob_sha[2:]
        try:
            if blob_file.exists():
                return zlib.decompress(blob_file.read_bytes())
        except Exception:
            logger.debug("Failed to read blob %s", blob_sha, exc_info=True)
        return None

    def delete_blob_if_unreferenced(self, blob_sha: str, rel_path: str | None = None) -> None:
        """Delete blob file if no other document references it.

        If a `repos_conn` was provided at construction time, use it to check
        the `documents` table for references. Otherwise attempt a best-effort
        file deletion.
        """
        try:
            ref_count = None
            if self.repos_conn is not None:
                try:
                    curs = self.repos_conn.cursor()
                    curs.execute("SELECT COUNT(*) FROM documents WHERE blob_sha = ?", (blob_sha,))
                    ref_count = curs.fetchone()[0]
                except Exception:
                    logger.debug("Failed to query documents table for blob refs %s", blob_sha, exc_info=True)
                    ref_count = None

            if ref_count is None:
                # Best-effort: delete file unconditionally
                blob_file = self.base_path / blob_sha[:2] / blob_sha[2:]
                try:
                    if blob_file.exists():
                        blob_file.unlink()
                except Exception:
                    logger.debug("Failed to delete blob file %s", blob_file, exc_info=True)
                return

            if ref_count == 0:
                blob_file = self.base_path / blob_sha[:2] / blob_sha[2:]
                try:
                    if blob_file.exists():
                        blob_file.unlink()
                except Exception:
                    logger.debug(
                        "Failed to delete blob file %s for %s",
                        blob_file,
                        rel_path,
                        exc_info=True,
                    )
        except Exception:
            logger.debug("Error in delete_blob_if_unreferenced for %s", blob_sha, exc_info=True)
