"""Trigram inverted-index wrapper using rocksdict or python-rocksdb.

Implements a minimal API compatible with the existing `sigil_mcp.indexer` helpers
so the later refactor can swap in this class with minimal changes.
"""

from __future__ import annotations

import os
import zlib
import json
import logging
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


# Try to import preferred backends in same order as indexer
try:
    from rocksdict import Rdict  # type: ignore
    ROCKSDICT_AVAILABLE = True
except Exception:
    Rdict: Any = None
    ROCKSDICT_AVAILABLE = False

try:
    import rocksdb as _rocksdb  # type: ignore
    ROCKSDB_AVAILABLE = True
except Exception:
    _rocksdb: Any = None
    ROCKSDB_AVAILABLE = False


class TrigramIndex:
    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self._backend: str = "none"
        self._rd: Any = None
        self._rb: Any = None
        self._init(path)

    def _init(self, path: Path) -> None:
        if ROCKSDICT_AVAILABLE:
            try:
                # Rdict is a callable mapping type; instantiate with path
                self._rd = Rdict(str(path))
                self._backend = "rocksdict"
                return
            except Exception:
                logger.exception("Failed to initialize rocksdict trigram store")
                raise
        if ROCKSDB_AVAILABLE:
            try:
                rb = cast(Any, _rocksdb)
                opts = rb.Options()
                opts.create_if_missing = True
                opts.max_background_jobs = 2
                opts.max_open_files = 500
                opts.write_buffer_size = 64 * 1024 * 1024
                opts.target_file_size_base = 64 * 1024 * 1024
                opts.compression = rb.CompressionType.lz4_compression
                table_opts = rb.BlockBasedTableOptions()
                table_opts.block_cache = rb.LRUCache(128 * 1024 * 1024)
                table_opts.filter_policy = rb.BloomFilterPolicy(10)
                opts.table_factory = rb.BlockBasedTableFactory(table_opts)
                self._rb = rb.DB(str(path), opts)
                self._backend = "rocksdb"
                return
            except Exception:
                logger.exception("Failed to initialize python-rocksdb trigram store")
                raise
        raise RuntimeError("No RocksDB backend available. Install rocksdict")

    # Serialization helpers borrowed from indexer
    @staticmethod
    def _serialize_doc_ids(doc_ids: set[int]) -> bytes:
        try:
            if not doc_ids:
                return zlib.compress(b"")
            ids = sorted(int(x) for x in doc_ids)
            max_id = max(ids)
            # Choose width: 4 bytes for <= uint32, else 8 bytes
            if max_id < (1 << 32):
                prefix = b"\x02"
                body = b"".join(int(i).to_bytes(4, "little") for i in ids)
            else:
                prefix = b"\x03"
                body = b"".join(int(i).to_bytes(8, "little") for i in ids)
            return zlib.compress(prefix + body)
        except Exception:
            logger.debug("Failed to serialize doc_ids", exc_info=True)
            return zlib.compress(b"")
        

    @staticmethod
    def _deserialize_doc_ids(blob: bytes) -> set[int]:
        try:
            raw = zlib.decompress(blob)
            if not raw:
                return set()
            # Legacy comma-separated text
            if raw.startswith(b"[") or b"," in raw:
                try:
                    text = raw.decode("utf-8")
                    # JSON array like: [1,2,3]
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return {int(x) for x in parsed}
                except Exception:
                    try:
                        # Fallback: comma-separated like "1,3,5"
                        if b"," in raw:
                            parts = text.split(",")
                            return {int(x.strip()) for x in parts if x.strip()}
                        return {int(x) for x in text.split("\n") if x}
                    except Exception:
                        return set()

            prefix = raw[:1]
            body = raw[1:]
            if prefix == b"\x02":
                # uint32 little-endian
                ids = [int.from_bytes(body[i : i + 4], "little") for i in range(0, len(body), 4)]
                return set(ids)
            elif prefix == b"\x03":
                ids = [int.from_bytes(body[i : i + 8], "little") for i in range(0, len(body), 8)]
                return set(ids)
            else:
                # Unknown format, attempt best-effort JSON parse
                try:
                    txt = raw.decode("utf-8")
                    parsed = json.loads(txt)
                    if isinstance(parsed, list):
                        return {int(x) for x in parsed}
                except Exception:
                    return set()
        except Exception:
            logger.debug("Failed to deserialize doc_ids", exc_info=True)
        return set()

    @staticmethod
    def serialize_trigram_set(trigrams: set[str]) -> bytes:
        try:
            return zlib.compress(json.dumps(sorted(trigrams), ensure_ascii=False).encode("utf-8"))
        except Exception:
            logger.debug("Failed to serialize trigram set", exc_info=True)
            return b""

    @staticmethod
    def deserialize_trigram_set(blob: bytes) -> set[str]:
        try:
            data = zlib.decompress(blob).decode("utf-8")
            try:
                parsed = json.loads(data)
                if isinstance(parsed, list):
                    return {str(t) for t in parsed}
            except Exception:
                return {t for t in data.split("\n") if t}
        except Exception:
            logger.debug("Failed to deserialize trigram set", exc_info=True)
        return set()

    def commit(self) -> None:
        if self._backend == "rocksdict" and self._rd is not None:
            try:
                if hasattr(self._rd, "flush_wal"):
                    self._rd.flush_wal()
                if hasattr(self._rd, "flush"):
                    self._rd.flush()
            except Exception:
                logger.debug("rocksdict flush failed", exc_info=True)

    def get_doc_ids(self, gram: str) -> set[int]:
        if self._backend != "rocksdict" or self._rd is None:
            return set()
        raw = self._rd.get(gram.encode(), None)
        if raw is None:
            return set()
        return self._deserialize_doc_ids(raw)

    def set_doc_ids(self, gram: str, doc_ids: set[int]) -> None:
        if not doc_ids:
            return self.delete(gram)
        serialized = self._serialize_doc_ids(doc_ids)
        if self._backend != "rocksdict" or self._rd is None:
            return
        retries = int(os.getenv("SIGIL_MCP_TRIGRAM_WRITE_RETRIES", "3"))
        backoff_ms = int(os.getenv("SIGIL_MCP_TRIGRAM_RETRY_BACKOFF_MS", "50"))
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                self._rd[gram.encode()] = serialized
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                logger.warning("rocksdict write attempt %d/%d failed for %s", attempt, retries, gram)
                if attempt < retries:
                    import time

                    time.sleep(backoff_ms / 1000.0)
        if last_exc is not None:
            logger.error("Failed to write trigram %s after %d attempts", gram, retries, exc_info=True)
            raise last_exc

    def delete(self, gram: str) -> None:
        if self._backend != "rocksdict" or self._rd is None:
            return
        try:
            del self._rd[gram.encode()]
        except KeyError:
            pass
        except Exception:
            logger.debug("Failed to delete trigram %s", gram, exc_info=True)

    def iter_items(self):
        if self._backend != "rocksdict" or self._rd is None:
            return []
        try:
            for k, v in self._rd.items():
                if isinstance(k, (bytes, bytearray)):
                    k0 = k.decode()
                else:
                    k0 = str(k)
                yield k0, self._deserialize_doc_ids(v)
        except Exception:
            logger.debug("Failed to iterate rocksdict items", exc_info=True)
            return []

    def count(self) -> int:
        if self._backend != "rocksdict" or self._rd is None:
            return 0
        try:
            cnt = len(self._rd)
            if cnt:
                return cnt
        except Exception:
            logger.debug("rocksdict len() failed", exc_info=True)
        try:
            return sum(1 for _ in self.iter_items())
        except Exception:
            logger.debug("Failed to count trigram items by iteration", exc_info=True)
            return 0

    def close(self) -> None:
        try:
            if self._rd is not None and hasattr(self._rd, "close"):
                self._rd.close()
        except Exception:
            logger.debug("Error closing rocksdict", exc_info=True)
        try:
            if self._rb is not None and hasattr(self._rb, "close"):
                self._rb.close()
        except Exception:
            logger.debug("Error closing rocksdb", exc_info=True)


