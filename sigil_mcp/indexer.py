# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Hybrid code indexing for Sigil MCP Server.

Combines trigram-based text search with symbol extraction for IDE-like features.
Designed to work well with ChatGPT and other AI assistants via MCP.
"""
# pyright: reportGeneralTypeIssues=false

import hashlib
import logging
import os
import re
# sqlite3 and subprocess are provided to storage/backends when needed; keep
# this module free of direct DB/process imports to avoid unused import issues.
import threading
import zlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from .analysis import (Symbol, SymbolExtractor, chunk_text, classify_path,
                       count_tokens, enforce_chunk_size_limits, hard_wrap,
                       is_jsonl_path, parse_jsonl_records)
from .ignore_utils import (is_ignored_by_gitignore, load_gitignore,
                           load_include_patterns, should_ignore)

# Trigram store requires rocksdict (RocksDB-backed Python bindings).
try:
    from rocksdict import Rdict  # type: ignore

    ROCKSDICT_AVAILABLE = True
except Exception:
    Rdict = None  # type: ignore
    ROCKSDICT_AVAILABLE = False

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
from .storage.blob import BlobStore
from .storage.metadata import MetadataStore
from .storage.symbols import SymbolStore
from .storage.trigram import TrigramIndex
from .storage.vector import VectorIndex

# Optional tokenizer support
try:
    import tiktoken  # type: ignore

    TIKTOKEN_AVAILABLE = True
except Exception:
    tiktoken = None  # type: ignore
    TIKTOKEN_AVAILABLE = False

# Tell the type-checker these optional third-party modules/objects are untyped
Rdict = cast(Any, Rdict)
lancedb = cast(Any, lancedb)
pa = cast(Any, pa)
tiktoken = cast(Any, tiktoken)

logger = logging.getLogger(__name__)

# Type alias for embedding function: takes sequence of texts, returns (N, dim) array
EmbeddingFn = Callable[[Sequence[str]], np.ndarray]

USE_LANCEDB_STUB = os.getenv("SIGIL_MCP_LANCEDB_STUB", "").lower() == "1"

# Heuristics to keep embedding workloads small and useful
DEFAULT_EMBED_CHUNK_LINES = 80
DEFAULT_EMBED_CHUNK_OVERLAP = 20

# Files/extensions/names we should never send to the embedder by default
EMBED_SKIP_EXTS = {".ipynb", ".jsonl"}
EMBED_SKIP_NAMES = {"package-lock.json", "Cargo.lock", ".coverage"}
DEFAULT_EMBED_MAX_CHARS = 4000
DEFAULT_EMBED_MAX_BYTES = 1_000_000
EMBED_SKIP_SUFFIXES = {".jsonl", ".log"}
EMBED_SKIP_FILENAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "cargo.lock",
    "poetry.lock",
    "pipfile.lock",
    "composer.lock",
    "gemfile.lock",
}
EMBED_SKIP_DIRS = {"coverage", "htmlcov", "test_logs", "output", "outputs", "backups"}


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

    def __init__(self, rows: list[dict], query_vec: np.ndarray | None):
        self._rows = list(rows or [])
        self._query_vec = query_vec.astype("float32") if query_vec is not None else None
        self._limit: int | None = None

    def metric(self, _m: str):
        # Stub: ignore metric selection
        return self

    def limit(self, n: int):
        self._limit = int(n)
        return self

    def _apply_filter(self) -> list[dict]:
        # Stub: no additional filters applied in this lightweight implementation
        return list(self._rows)

    def to_list(self) -> list[dict]:
        rows = self._apply_filter()
        scored: list[tuple[float, dict]] = []
        for row in rows:
            vec = np.asarray(row.get("vector", []), dtype="float32")
            if self._query_vec is not None and vec.shape == self._query_vec.shape:
                score = float(np.dot(vec, self._query_vec))
                distance = float(1.0 - score)
            else:
                score = 0.0
                distance = 1.0
            row_copy = dict(row)
            row_copy["_distance"] = distance
            scored.append((score, row_copy))

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

    def _normalize_record(self, record: Any) -> dict:
        if hasattr(record, "model_dump"):
            data = cast(dict, record.model_dump())
        elif hasattr(record, "dict"):
            data = cast(dict, record.dict())  # type: ignore[attr-defined]
        else:
            data = dict(record)  # type: ignore[arg-type]

        vec = data.get("vector")
        if vec is not None and hasattr(vec, "tolist"):
            vec = cast(list, vec.tolist())
        data["vector"] = vec
        return data

    def count_rows(self) -> int:
        return len(self.rows)

    def to_arrow(self):
        return _InMemoryArrowTable(self.rows)

    def to_list(self, limit: int | None = None) -> list[dict]:
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

    def create_table(
        self, name: str, schema=None, mode: str | None = None, **_kwargs
    ) -> InMemoryLanceTable:
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


_STUB_LANCEDB_REGISTRY: dict[str, InMemoryLanceDB] = {}


@dataclass
class SearchResult:
    """Represents a search result."""

    repo: str
    path: str
    line: int
    text: str
    doc_id: str
    symbol: Symbol | None = None


class SigilIndex:
    """Hybrid index supporting both text and symbol search."""

    def __init__(
        self, index_path: Path, embed_fn: EmbeddingFn | None = None, embed_model: str = "local"
    ):
        self.index_path = index_path

        self.embed_fn = embed_fn
        self.embed_model = embed_model
        self._embedding_init_failed = False
        self.use_lancedb_stub = USE_LANCEDB_STUB

        config = get_config()
        # Keep config available for runtime decisions (ignore patterns, per-repo options)
        self.config = config
        self.embedding_dimension = config.embeddings_dimension
        self.embedding_provider = config.embeddings_provider
        self.allow_vector_schema_overwrite = config.index_allow_vector_schema_overwrite
        # Base directory under which per-repo LanceDB directories will live.
        # Previously we used a single shared LanceDB path; migrate to
        # per-repo directories located under this base path.
        if config.index_path == self.index_path:
            self.base_lance_dir = Path(config.lance_dir)
        else:
            self.base_lance_dir = self.index_path / "lancedb"

        # Cache of per-repo LanceDB connections / tables
        self._repo_lance_dbs: dict[str, object] = {}
        self._repo_vectors: dict[str, object] = {}
        # Separate locks: DB/trigram metadata vs embedding calls
        self._db_lock = threading.RLock()
        self._embed_lock = threading.RLock()
        # Backward compatibility with older callers that referenced _lock
        self._lock = self._db_lock

        # Backwards-compatible single global attributes (may remain None)
        self.lance_db: Any = None
        self.lance: Any = None
        self.lance_db_path = self.base_lance_dir
        self.vector_table_name = "code_vectors"
        self.vectors: Any = None
        self.lancedb_available = LANCEDB_AVAILABLE or self.use_lancedb_stub
        self._vector_index_stale = False

        # Helper: a small lock scoped to per-repo lance initialization
        self._lance_init_lock = threading.RLock()

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
            # Initialize VectorIndex wrapper (thin abstraction over LanceDB or stub)
            try:
                self.vector_index = VectorIndex(self.base_lance_dir, self.embedding_dimension, use_stub=self.use_lancedb_stub or not LANCEDB_AVAILABLE)
                # Expose compatibility attributes
                self.lance_db = getattr(self.vector_index, "_db", None)
                self.vectors = self.vector_index.table(self.vector_table_name) or self.vectors
            except Exception:
                logger.debug("Failed to initialize VectorIndex wrapper", exc_info=True)
        # Wrap embed_fn to apply batching/bucketing heuristics to avoid padding blowup
        # Preserve the original function object on `embed_fn` for callers/tests
        # while using `_wrapped_embed_fn` internally for batching behavior.
        if self.embed_fn is not None:
            self._raw_embed_fn = self.embed_fn
            try:
                if self.embedding_provider != "llamacpp":
                    self._wrapped_embed_fn = self._wrap_embed_fn(self._raw_embed_fn)
                else:
                    # Skip bucketing for llama.cpp to avoid redundant tokenization
                    self._wrapped_embed_fn = self._raw_embed_fn
            except Exception:
                # If wrapping fails, fall back to raw function
                self._wrapped_embed_fn = self._raw_embed_fn
        else:
            self._raw_embed_fn = None
            self._wrapped_embed_fn = None
            self._code_chunk_model = None
            self.vectors = None

        self._vector_index_enabled = self._embeddings_active and self.vectors is not None
        self._log_vector_index_status()

        # Trigram postings are stored via rocksdict/rocksdb and metadata is in SQLite.
        self._trigram_backend: str = "unknown"
        self._rocksdict_trigrams: Any = None

        # Initialize metadata store (SQLite wrapper)
        try:
            self.metadata = MetadataStore(self.index_path / "repos.db")
            self.repos_db = self.metadata.conn
        except Exception:
            logger.exception("Failed to initialize metadata store")
            raise

        # Blob store (compressed blob files)
        try:
            self.blob_store = BlobStore(self.index_path / "blobs", repos_conn=self.repos_db)
        except Exception:
            logger.exception("Failed to initialize blob store")
            self.blob_store = None

        # Symbol extraction & storage helpers
        try:
            self.symbol_extractor = SymbolExtractor()
            self.symbol_store = SymbolStore(self.repos_db)
        except Exception:
            logger.debug("Failed to initialize symbol extractor/store", exc_info=True)
            self.symbol_extractor = SymbolExtractor()
            self.symbol_store = None

        # Initialize trigram store (rocksdict or python-rocksdb wrapper)
        try:
            trigram_path = self.index_path / "trigrams.rocksdb"
            trigram_path.mkdir(parents=True, exist_ok=True)
            self._trigram_index = TrigramIndex(trigram_path)
            # Expose compatibility attributes used elsewhere in code
            if getattr(self._trigram_index, "_backend", None) == "rocksdict":
                self._rocksdict_trigrams = getattr(self._trigram_index, "_rd", None)
                self._trigram_backend = "rocksdict"
            else:
                self._rocksdb_trigrams = getattr(self._trigram_index, "_rb", None)
                self._trigram_backend = "rocksdb" if getattr(self._trigram_index, "_rb", None) is not None else "unknown"
            logger.info("Using %s trigram store at %s", self._trigram_backend, trigram_path)
        except Exception as exc:
            logger.exception("Failed to initialize trigram index")
            raise RuntimeError(
                "trigram initialization failed - RocksDB trigrams are required"
            ) from exc

        self._init_schema()
        self._check_vector_repo_alignment()

    def _call_embed(self, texts: Sequence[str]) -> np.ndarray:
        """Invoke the embedding function, preferring the internal wrapped/batched version.

        Keeps `self.embed_fn` pointing at the original function object for tests that
        assert identity, while using `_wrapped_embed_fn` for actual embedding calls.
        """
        fn = getattr(self, "_wrapped_embed_fn", None) or getattr(self, "embed_fn", None)
        if fn is None:
            return np.zeros((0, self.embedding_dimension), dtype="float32")
        try:
            if self.embedding_provider == "llamacpp":
                with self._embed_lock:
                    return fn(list(texts))
            return fn(list(texts))
        except Exception:
            # Fail-open: return empty array on error
            return np.zeros((0, self.embedding_dimension), dtype="float32")

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

        # Ensure base directory for per-repo lance DBs exists and has sane perms
        created = False
        try:
            if not self.base_lance_dir.exists():
                self.base_lance_dir.mkdir(parents=True, exist_ok=True)
                created = True
            else:
                self.base_lance_dir.mkdir(parents=True, exist_ok=True)

            if created:
                try:
                    mode = self.index_path.stat().st_mode & 0o777
                    os.chmod(self.base_lance_dir, mode)
                except Exception:
                    logger.exception(
                        "Failed to apply index directory permissions to LanceDB base path %s",
                        self.base_lance_dir,
                    )
        except Exception:
            logger.exception("Failed to ensure base LanceDB directory %s", self.base_lance_dir)

    def _get_repo_lance_path(self, repo_name: str) -> Path:
        """Compute the LanceDB directory path for a given repo name."""
        return (self.base_lance_dir / repo_name).resolve()

    def _get_repo_name_for_id(self, repo_id: int) -> str | None:
        with self._db_lock:
            cur = self.repos_db.cursor()
            cur.execute("SELECT name FROM repos WHERE id = ?", (repo_id,))
            row = cur.fetchone()
        return row[0] if row else None

    def _get_repo_lance_and_vectors(self, repo_name: str):
        """Return a (lance_db, vectors_table) tuple for the given repo, creating
        the LanceDB and table if necessary. Caches connections in-memory."""
        with self._lance_init_lock:
            if repo_name in self._repo_vectors and repo_name in self._repo_lance_dbs:
                return self._repo_lance_dbs[repo_name], self._repo_vectors[repo_name]

            path = self._get_repo_lance_path(repo_name)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.exception("Failed to create LanceDB path for repo %s at %s", repo_name, path)

            # Initialize stub or real LanceDB for this repo via VectorIndex wrapper
            try:
                repo_db = self.vector_index.get_repo_db(path)
            except Exception:
                logger.exception("Failed to initialize per-repo LanceDB for %s at %s", repo_name, path)
                repo_db = None

            repo_table = None
            if repo_db is not None:
                try:
                    table_names = set(repo_db.table_names())
                    target_table = self.vector_table_name
                    if target_table in table_names:
                        repo_table = repo_db.open_table(target_table)
                    elif "code_chunks" in table_names:
                        target_table = "code_chunks"
                        repo_table = repo_db.open_table(target_table)
                    else:
                        try:
                            # Use VectorIndex helper to create per-repo table
                            repo_table = self.vector_index.create_table_for_repo(
                                repo_db, target_table, get_code_chunk_model(self.embedding_dimension)
                            )
                        except Exception:
                            repo_table = None
                except Exception:
                    logger.exception(
                        "Failed to open/create vector table for repo %s at %s", repo_name, path
                    )
                    repo_table = None

            self._repo_lance_dbs[repo_name] = repo_db
            self._repo_vectors[repo_name] = repo_table
            return repo_db, repo_table

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

    def _wrap_embed_fn(self, fn: EmbeddingFn) -> EmbeddingFn:
        """Wrap a provider embed function to apply simple length bucketing and batching.

        This reduces padding waste by grouping similarly-sized texts together
        before calling the underlying provider.
        """

        def wrapped(texts: Sequence[str]) -> np.ndarray:
            # Use tokenizer-based token counting when available for accurate bucketing
            thresholds = list(self._get_bucket_thresholds())
            # buckets is list of lists (preserve order of thresholds)
            buckets: list[list[tuple[str, int]]] = [[] for _ in range(len(thresholds) + 1)]

            def count_tokens(s: str) -> int:
                # Prefer tiktoken if available and a model is known
                try:
                    if TIKTOKEN_AVAILABLE and self.embed_model:
                        tk = cast(Any, tiktoken)
                        try:
                            enc = tk.encoding_for_model(self.embed_model)
                        except Exception:
                            enc = tk.get_encoding("cl100k_base")
                        return len(enc.encode(s))
                except Exception:
                    pass
                # Fallback: whitespace heuristic (words)
                return max(1, len(s.split()))

            for t in texts:
                toks = count_tokens(t)
                placed = False
                for i, th in enumerate(thresholds):
                    if toks <= th:
                        buckets[i].append((t, toks))
                        placed = True
                        break
                if not placed:
                    buckets[-1].append((t, toks))

            results: list[np.ndarray] = []

            def process_bucket(items: list[tuple[str, int]]):
                batch_size = 64
                texts_only = [i[0] for i in items]
                for i in range(0, len(texts_only), batch_size):
                    batch = texts_only[i : i + batch_size]
                    out = fn(list(batch))
                    arr = np.asarray(out)
                    # ensure 2D array (N, dim)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    for row in arr:
                        results.append(row)

            for bucket in buckets:
                if bucket:
                    process_bucket(bucket)

            if results:
                return np.asarray(results, dtype="float32")
            # default empty array with correct dimensionality
            return np.zeros((0, self.embedding_dimension), dtype="float32")

        return wrapped

    def _get_bucket_thresholds(self) -> list[int]:
        cfg = get_config()
        try:
            vals = list(cfg.embeddings_bucket_thresholds)
            # ensure ints and sorted
            vals = sorted(int(v) for v in vals)
            return vals
        except Exception:
            return [256, 512, 1024, 2048]

    def _hard_wrap(self, text: str) -> list[str]:
        """Compatibility wrapper around analysis.hard_wrap."""
        return hard_wrap(text)

    def _is_jsonl_path(self, p: str | Path) -> bool:
        """Compatibility wrapper around analysis.is_jsonl_path."""
        return is_jsonl_path(p)

    def _enforce_chunk_size_limits(
        self, chunks: list[tuple[int, int, int, str]]
    ) -> list[tuple[int, int, int, str]]:
        """Ensure chunks are not excessively large by delegating to analysis helpers."""
        return enforce_chunk_size_limits(
            chunks,
            embedding_provider=self.embedding_provider,
            embed_model=self.embed_model,
        )

    def _count_tokens(self, s: str) -> int:
        """Utility to count tokens for a string using shared analysis helper."""
        return count_tokens(
            s, embed_model=self.embed_model, embedding_provider=self.embedding_provider
        )

    def _parse_jsonl_records(self, text: str, include_solution: bool | None = None) -> list[str]:
        """Compatibility wrapper to route parsing to analysis helpers."""
        return parse_jsonl_records(text, include_solution=include_solution)

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

    @staticmethod
    def _classify_path(rel_path: str, sample_text: str | None = None) -> dict[str, object]:
        return classify_path(rel_path, sample_text=sample_text)

    def _log_vector_index_status(self, context: str = "startup") -> None:
        """Log current vector index availability and size."""
        # If we have per-repo vector tables, log per-repo status; otherwise
        # fall back to global table logging.
        try:
            if self._repo_vectors:
                for repo_name, table in self._repo_vectors.items():
                    try:
                        count = int(cast(Any, table).count_rows()) if table is not None else -1
                    except Exception:
                        logger.debug("Failed to count rows for repo %s", repo_name, exc_info=True)
                        count = -1
                    path = self._get_repo_lance_path(repo_name)
                    logger.info(
                        "Vector index ready (%s) for repo %s: %s indexed chunks at %s",
                        context,
                        repo_name,
                        count,
                        path,
                    )
                return

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
        except Exception:
            logger.exception("Error logging vector index status")

    def _sample_vector_repo_ids(self, limit: int = 200) -> set[str]:
        """Read a small sample of repo_ids from LanceDB for sanity checks."""
        repo_ids: set[str] = set()

        # Prefer sampling from per-repo tables
        try:
            if self._repo_vectors:
                for repo_name, table in self._repo_vectors.items():
                    if table is None:
                        continue
                    try:
                        if hasattr(table, "to_list"):
                            rows = cast(Any, table).to_list(limit=limit)
                        else:
                            t = cast(Any, table).head(limit)
                            rows = t.to_pylist() if t is not None else []
                    except Exception:
                        logger.debug("Failed to sample rows from repo %s", repo_name, exc_info=True)
                        continue
                    for r in rows:
                        if r.get("repo_id") is not None:
                            repo_ids.add(str(r.get("repo_id")))
                return repo_ids

            # Fallback: sample from global table
            if self.vectors is None:
                return set()
            if hasattr(self.vectors, "to_list"):
                rows = cast(Any, self.vectors).to_list(limit=limit)
            else:
                table = cast(Any, self.vectors).head(limit)
                rows = table.to_pylist() if table is not None else []
        except Exception:
            logger.exception("Failed to sample repo_ids from vector table")
            return set()

        return {str(r.get("repo_id")) for r in rows if r.get("repo_id") is not None}

    def _check_vector_repo_alignment(self) -> None:
        """
        Detect repo-id drift between LanceDB and repos.db.

        If repo IDs in the vector table do not overlap any known repos,
        semantic search is marked stale to avoid silently returning no results.
        """

        if not self._vector_index_enabled:
            return

        with self._db_lock:
            repo_rows = list(self.repos_db.execute("SELECT id FROM repos"))
        repo_ids = {str(row[0]) for row in repo_rows}
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
        self.repos_db.execute(
            """
            CREATE TABLE IF NOT EXISTS repos (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                path TEXT,
                indexed_at TEXT
            )
        """
        )

        self.repos_db.execute(
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

        self.repos_db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_doc_path
            ON documents(repo_id, path)
        """
        )

        # Ensure blob_sha is indexed (but not unique) for reuse/lookups
        self.repos_db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_doc_blob_sha
            ON documents(blob_sha)
        """
        )

        # Symbol index for IDE-like features
        self.repos_db.execute(
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

        self.repos_db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_symbol_name
            ON symbols(name)
        """
        )

        self.repos_db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_symbol_kind
            ON symbols(kind)
        """
        )

        # Remove legacy embeddings table now that vectors are stored in LanceDB
        try:
            self.repos_db.execute("DROP TABLE IF EXISTS embeddings")
        except Exception:
            logger.debug("Skipping removal of legacy embeddings table", exc_info=True)

        # rocksdict-backed trigram store uses its own key/value path; nothing to initialize here.
        self.repos_db.commit()
        # Ensure vector metadata columns exist for backward-compatible upgrades
        try:
            cur = self.repos_db.cursor()
            cur.execute("PRAGMA table_info(documents)")
            cols = {row[1] for row in cur.fetchall()}
            if "vector_indexed_at" not in cols:
                try:
                    self.repos_db.execute("ALTER TABLE documents ADD COLUMN vector_indexed_at TEXT")
                except Exception:
                    # If ALTER fails for any reason, continue; column is non-critical
                    logger.debug(
                        "Could not add vector_indexed_at column to documents table",
                        exc_info=True,
                    )
            if "vector_index_error" not in cols:
                try:
                    self.repos_db.execute(
                        "ALTER TABLE documents ADD COLUMN vector_index_error TEXT"
                    )
                except Exception:
                    logger.debug(
                        "Could not add vector_index_error column to documents table",
                        exc_info=True,
                    )
            try:
                self.repos_db.commit()
            except Exception:
                pass
        except Exception:
            # Best-effort migration; don't block startup on metadata columns
            logger.debug("Skipping vector metadata migration check", exc_info=True)
        # Note: Do not alter the repos schema here to maintain compatibility with
        # test expectations that assert a minimal repos table (id, name, path, indexed_at).
        # If per-repo embedding options are added in the future, handle migrations
        # externally or via a dedicated migration step.

    def _get_repo_include_solution(self, repo_id: int) -> bool | None:
        """Return per-repo embeddings.include_solution setting (True/False) or None if unset.

        This reads the `repos` table for the optional `embeddings_include_solution` column
        introduced in schema migrations. Returns None when repository has no explicit
        per-repo setting so callers can fall back to global config.
        """
        try:
            with self._db_lock:
                cur = self.repos_db.cursor()
                cur.execute(
                    "SELECT embeddings_include_solution FROM repos WHERE id = ?", (repo_id,)
                )
                row = cur.fetchone()
            if not row:
                return None
            val = row[0]
            if val is None:
                return None
            return bool(val)
        except Exception:
            return None

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
        try:
            # Normalize paths to avoid relative/absolute mismatches.
            # Some callers pass relative paths (e.g., admin API requests); always
            # resolve them under the configured repo root and ensure they do not
            # escape the repository.
            repo_path = Path(repo_path).resolve()
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = (repo_path / file_path).resolve()
            else:
                file_path = file_path.resolve()

            try:
                file_path.relative_to(repo_path)
            except ValueError as exc:
                raise ValueError(f"{file_path} is not in the subpath of {repo_path}") from exc

            # Get or create repo entry
            with self._db_lock:
                cursor = self.repos_db.cursor()
                cursor.execute(
                    "INSERT OR IGNORE INTO repos (name, path, indexed_at) " "VALUES (?, ?, ?)",
                    (repo_name, str(repo_path), datetime.now().isoformat()),
                )
                cursor.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
                repo_id = cursor.fetchone()[0]

            # Determine language
            file_extensions = {
                ".py": "python",
                ".rs": "rust",
                ".js": "javascript",
                ".ts": "typescript",
                ".java": "java",
                ".go": "go",
                ".cpp": "cpp",
                ".c": "c",
                ".h": "c",
                ".hpp": "cpp",
                ".rb": "ruby",
                ".php": "php",
                ".cs": "csharp",
                ".sh": "shell",
                ".toml": "toml",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".json": "json",
                ".md": "markdown",
            }
            ext = file_path.suffix.lower()
            language = file_extensions.get(ext, "unknown")

            # Index the specific file (metadata + symbols)
            result = self._index_file(repo_id, repo_name, repo_path, file_path, language)
            if not result:
                with self._db_lock:
                    try:
                        self.repos_db.commit()
                    except Exception:
                        logger.debug(
                            "Failed to commit after no-op index_file for %s",
                            file_path,
                            exc_info=True,
                        )
                return False

            res = cast(dict, result)
            doc_id = int(res.get("doc_id", 0))
            rel_path = res.get("rel_path", file_path.as_posix())
            embed_allowed = bool(res.get("embed_allowed", True))
            embeddings: np.ndarray | None = None
            chunks: list[tuple[int, int, int, str]] = []

            # Embedding path: only when vector store and embedding fn are available
            if self.vectors is not None and self.embed_fn is not None and embed_allowed:
                # Special-case JSONL: chunk by record (per-line), extracting useful fields
                if self._is_jsonl_path(file_path):
                    include_sol = self._get_repo_include_solution(repo_id)
                    records = self._parse_jsonl_records(
                        res.get("text", ""), include_solution=include_sol
                    )
                    chunks = [(i, 1, 1, r) for i, r in enumerate(records)] if records else []
                else:
                    chunks = self._chunk_text(res.get("text", ""))
                # Enforce hard caps on chunk size to avoid runaway chunks
                chunks = self._enforce_chunk_size_limits(chunks)
                if chunks:
                    try:
                        embeddings = self._call_embed([c[3] for c in chunks])
                    except Exception as e:
                        logger.exception("Embedding failed for %s: %s", file_path, e)
                        with self._db_lock:
                            try:
                                self.repos_db.execute(
                                    "UPDATE documents SET vector_index_error = ? WHERE id = ?",
                                    (str(e)[:1024], doc_id),
                                )
                                self.repos_db.commit()
                            except Exception:
                                logger.debug(
                                    "Failed to record vector_index_error for doc %s",
                                    doc_id,
                                    exc_info=True,
                                )
                        embeddings = None
            else:
                # If embedding was skipped by policy, mark it explicitly.
                if not embed_allowed:
                    with self._db_lock:
                        try:
                            self.repos_db.execute(
                                "UPDATE documents SET vector_index_error = ? WHERE id = ?",
                                ("embed_skipped_by_policy", doc_id),
                            )
                            self.repos_db.commit()
                        except Exception:
                            logger.debug(
                                "Failed to record embed_skipped flag for doc %s",
                                doc_id,
                                exc_info=True,
                            )
                else:
                    # Vector store or embed function not available — record reason
                    reason = None
                    if self.vectors is None:
                        reason = "vector_store_unavailable"
                    elif self.embed_fn is None:
                        reason = "embeddings_unavailable"
                    if reason is not None:
                        with self._db_lock:
                            try:
                                self.repos_db.execute(
                                    "UPDATE documents SET vector_index_error = ? WHERE id = ?",
                                    (reason, doc_id),
                                )
                                self.repos_db.commit()
                            except Exception:
                                logger.debug(
                                    "Failed to record vector_index_error for doc %s",
                                    doc_id,
                                    exc_info=True,
                                )

            if embeddings is not None and chunks:
                self._index_file_vectors(
                    repo_id,
                    doc_id,
                    rel_path,
                    chunks,
                    embeddings,
                )

            # Rebuild trigrams for this file
            self._update_trigrams_for_file(repo_id, repo_path, file_path)
            with self._db_lock:
                self.repos_db.commit()
                logger.info(f"Re-indexed {file_path.name} in {repo_name}")
                return True
        except Exception:
            # Catch-all to ensure the try-block opened at the start of this
            # function is properly closed and to provide a safe failure path.
            logger.exception("Failed to index file %s in %s", file_path, repo_name)
            return False

    def _update_trigrams_for_file(self, repo_id: int, repo_path: Path, file_path: Path):
        """Update trigrams for a specific file."""
        # Calculate relative path (same way _index_file does)
        rel_path = file_path.relative_to(repo_path).as_posix()

        # Get document ID and blob SHA
        with self._db_lock:
            cursor = self.repos_db.cursor()
            cursor.execute(
                "SELECT id, blob_sha FROM documents WHERE repo_id = ? AND path = ?",
                (repo_id, rel_path),
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

        text = content.decode("utf-8", errors="replace").lower()
        new_trigrams = self._extract_trigrams(text)
        prior_trigrams = self._load_doc_trigrams(doc_id)
        logger.debug(
            "_update_trigrams_for_file: repo_id=%s doc_id=%s blob_sha=%s trigrams=%s",
            repo_id,
            doc_id,
            blob_sha,
            len(new_trigrams),
        )

        stale_trigrams = prior_trigrams.difference(new_trigrams)
        if stale_trigrams:
            self._remove_doc_from_trigram_postings(doc_id, stale_trigrams)

        self._add_doc_to_trigram_postings(doc_id, new_trigrams)
        self._store_doc_trigrams(doc_id, new_trigrams)

    def index_repository(
        self, repo_name: str, repo_path: Path, force: bool = False
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
        logger.info(f"Indexing repository: {repo_name} at {repo_path}")

        start_time = datetime.now()
        stats: dict[str, int] = {
            "files_indexed": 0,
            "symbols_extracted": 0,
            "trigrams_built": 0,
            "bytes_indexed": 0,
        }

        # Register or update repo
        with self._db_lock:
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
                if old_doc_ids:
                    logger.info(
                        "Force rebuild: clearing old index data for %s (%s docs)",
                        repo_name,
                        len(old_doc_ids),
                    )
                    # Use SymbolStore to delete symbols for each old doc to keep
                    # symbol persistence logic centralized.
                    if getattr(self, "symbol_store", None) is not None:
                        for did in old_doc_ids:
                            self.symbol_store.delete_symbols_for_doc(did)
                    else:
                        placeholders = ",".join("?" for _ in old_doc_ids)
                        cursor.execute(
                            f"DELETE FROM symbols WHERE doc_id IN ({placeholders})",
                            tuple(old_doc_ids),
                        )
                    cursor.execute("DELETE FROM documents WHERE repo_id = ?", (repo_id,))
                    # Remove only this repo's postings from trigram index
                    self._remove_trigrams_for_doc_ids(old_doc_ids)
                else:
                    logger.debug(
                        "Force rebuild: no existing index data for %s (0 docs)",
                        repo_name,
                    )

        # Index all files (parallel CPU prep; DB writes stay serialized by locks in _index_file)
        file_extensions = {
            ".py": "python",
            ".rs": "rust",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".sh": "shell",
            ".toml": "toml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".md": "markdown",
        }

        file_paths: list[Path] = []
        total_scanned = 0
        skipped = 0
        skipped_examples: list[str] = []
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            total_scanned += 1
            try:
                skip = self._should_skip(file_path, repo_path)
            except Exception:
                skip = False
            if skip:
                skipped += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append(str(file_path))
                continue
            file_paths.append(file_path)

        logger.info(
            "Index scan summary: scanned=%s skipped=%s included=%s sample_skipped=%s",
            total_scanned,
            skipped,
            len(file_paths),
            skipped_examples,
        )
        # Report top-level directory breakdown to help tune ignore patterns
        try:
            by_top: dict[str, int] = {}
            repo_root = Path(repo_path).resolve()
            for p in file_paths:
                try:
                    rel = p.resolve().relative_to(repo_root)
                    top = rel.parts[0] if rel.parts else "."
                except Exception:
                    top = str(p.parent)
                by_top[top] = by_top.get(top, 0) + 1
            # Log top 10 contributors
            top_items = sorted(by_top.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info("Index included files by top-level dir: %s", top_items)
        except Exception:
            pass

        worker_env = os.getenv("SIGIL_MCP_INDEX_WORKERS")
        if worker_env:
            try:
                max_workers = max(1, int(worker_env))
            except Exception:
                max_workers = max(1, (os.cpu_count() or 2) - 1)
        else:
            max_workers = max(1, (os.cpu_count() or 2) - 1)

        logger.info(
            "Indexing %s files for %s with %s workers", len(file_paths), repo_name, max_workers
        )

        if max_workers == 1 or len(file_paths) <= 1:
            for file_path in file_paths:
                ext = file_path.suffix.lower()
                language = file_extensions.get(ext, "unknown")
                file_stats = self._index_file(repo_id, repo_name, repo_path, file_path, language)
                if file_stats:
                    stats["files_indexed"] += 1
                    stats["symbols_extracted"] += cast(int, file_stats.get("symbols", 0))
                    stats["bytes_indexed"] += cast(int, file_stats.get("bytes", 0))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for file_path in file_paths:
                    ext = file_path.suffix.lower()
                    language = file_extensions.get(ext, "unknown")
                    futures.append(
                        executor.submit(
                            self._index_file,
                            repo_id,
                            repo_name,
                            repo_path,
                            file_path,
                            language,
                        )
                    )

                for fut in as_completed(futures):
                    try:
                        file_stats = fut.result()
                    except Exception:
                        logger.exception("Failed indexing file in parallel executor")
                        continue
                    if file_stats:
                        stats["files_indexed"] += 1
                        stats["symbols_extracted"] += cast(int, file_stats.get("symbols", 0))
                        stats["bytes_indexed"] += cast(int, file_stats.get("bytes", 0))

        with self._db_lock:
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
        self, repo_id: int, repo_name: str, repo_root: Path, file_path: Path, language: str
    ) -> dict[str, object] | None:
        """Index a single file."""
        try:
            content = file_path.read_bytes()
            raw_text = content.decode("utf-8", errors="replace")
            text = raw_text
            # Determine whether to allow embedding for this file
            ext = file_path.suffix.lower()
            name = file_path.name
            embed_allowed = True
            if ext in EMBED_SKIP_EXTS or name in EMBED_SKIP_NAMES:
                embed_allowed = False

            # Don't embed backup/timestamped files even if they appear in the documents table
            try:
                nl = name.lower()
                if ".backup" in nl or nl.endswith(".bak") or nl.endswith("~"):
                    embed_allowed = False
            except Exception:
                pass

            # Special-case: notebooks — extract cell sources (code + markdown) for embeddings
            if ext == ".ipynb":
                try:
                    import json as _json

                    nb = _json.loads(raw_text)
                    parts = []
                    for cell in nb.get("cells", []):
                        if cell.get("cell_type") in ("markdown", "code"):
                            src = "".join(cell.get("source", []) or [])
                            if src and src.strip():
                                parts.append(src)
                    cleaned = "\n\n".join(parts)
                    if cleaned.strip():
                        text = cleaned
                        embed_allowed = True
                    else:
                        embed_allowed = False
                except Exception:
                    embed_allowed = False

            # Special-case: JSONL with very long lines — skip embedding unless per-record indexing is implemented
            if self._is_jsonl_path(file_path) and raw_text:
                try:
                    # If any line is extremely long, avoid embedding whole file
                    for line in raw_text.splitlines():
                        if len(line) > 10000:
                            embed_allowed = False
                            break
                except Exception:
                    embed_allowed = False
            blob_sha = hashlib.sha256(content).hexdigest()
            rel_path = file_path.relative_to(repo_root).as_posix()
            # Extract symbols before acquiring DB lock so we don't hold it during ctags
            try:
                symbols = (
                    self.symbol_extractor.extract(file_path, language)
                    if getattr(self, "symbol_extractor", None) is not None
                    else []
                )
            except Exception:
                logger.debug("Symbol extraction failed", exc_info=True)
                symbols = []

            with self._db_lock:
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
                    self._delete_symbols_and_embeddings_for_doc(existing_doc_id, repo_id, rel_path)
                    self._remove_trigrams_for_doc_ids({existing_doc_id})
                    cursor.execute("DELETE FROM documents WHERE id = ?", (existing_doc_id,))

                # Store document metadata (repo/path scoped; blob_sha can be reused across repos)
                # Remove any additional stale rows for this repo/path just in case
                try:
                    cursor.execute(
                        "SELECT id FROM documents WHERE repo_id = ? AND path = ?",
                        (repo_id, rel_path),
                    )
                    rows = cursor.fetchall()
                    for (old_doc_id,) in rows:
                        self._delete_symbols_and_embeddings_for_doc(old_doc_id, repo_id, rel_path)
                        self._remove_trigrams_for_doc_ids({old_doc_id})
                        cursor.execute("DELETE FROM documents WHERE id = ?", (old_doc_id,))
                except Exception:
                    logger.exception("Failed to cleanup old document rows for %s", rel_path)

                cursor.execute(
                    """
                    INSERT INTO documents (repo_id, path, blob_sha, size, language)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (repo_id, rel_path, blob_sha, len(content), language),
                )
                doc_id = cursor.lastrowid

            # Store blob content (compressed)
                # Delegate blob write to BlobStore
                try:
                    if hasattr(self, "blob_store") and self.blob_store is not None:
                        self.blob_store.write_blob(blob_sha, content)
                    else:
                        blob_dir = self.index_path / "blobs" / blob_sha[:2]
                        blob_dir.mkdir(parents=True, exist_ok=True)
                        blob_file = blob_dir / blob_sha[2:]
                        if not blob_file.exists():
                            blob_file.write_bytes(zlib.compress(content))
                except Exception:
                    logger.debug("Failed to write blob %s", blob_sha, exc_info=True)

            # Store extracted symbols using the SymbolStore wrapper (if available)
            if symbols and getattr(self, "symbol_store", None) is not None:
                with self._db_lock:
                    self.symbol_store.store_symbols_for_doc(doc_id, symbols)

            return {
                "symbols": len(symbols),
                "bytes": int(len(content)),
                "doc_id": doc_id,
                "rel_path": rel_path,
                "text": text,
                "embed_allowed": embed_allowed,
            }

        except Exception as e:
            logger.warning(f"Error indexing {file_path}: {e}")
            return None



    def _build_trigram_index(self, repo_id: int) -> int:
        """Build trigram index for a repository's documents.

        Optimized approach: build trigram -> doc_id map in-memory, then
        batch-merge existing postings from the trigrams DB and write using
        executemany to reduce Python/SQLite round-trips and CPU overhead.
        """
        start = perf_counter()
        with self._db_lock:
            cursor = self.repos_db.cursor()
            docs = list(
                cursor.execute("SELECT id, blob_sha FROM documents WHERE repo_id = ?", (repo_id,))
            )

        trigram_map: dict[str, set[int]] = defaultdict(set)
        doc_count = 0
        for doc_id, blob_sha in docs:
            doc_count += 1
            content = self._read_blob(blob_sha)
            if not content:
                continue
            text = content.decode("utf-8", errors="replace").lower()
            doc_trigrams = self._extract_trigrams(text)
            for tg in doc_trigrams:
                trigram_map[tg].add(doc_id)
            # Persist per-doc trigram cache for fast removals later
            try:
                self._store_doc_trigrams(doc_id, doc_trigrams)
            except Exception:
                logger.debug(
                    "_build_trigram_index: failed to store doc trigrams for %s",
                    doc_id,
                    exc_info=True,
                )

        mid = perf_counter()
        logger.info(
            "_build_trigram_index: repo_id=%s docs=%s unique_trigrams=%s build_time=%.2fs",
            repo_id,
            doc_count,
            len(trigram_map),
            mid - start,
        )

        # Merge with any existing postings and write into the selected backend.
        grams = list(trigram_map.keys())
        if grams:
            # For Rocks-backed stores we use the K/V API to read/merge/write per-gram.
            for gram in grams:
                try:
                    existing_ids = self._trigram_get_doc_ids(gram)
                    doc_ids = trigram_map[gram].union(existing_ids)
                    # Use helper which will route to the proper backend
                    self._trigram_set_doc_ids(gram, doc_ids)
                except Exception:
                    logger.error("Failed to merge/write trigram %s", gram, exc_info=True)

        # Commit once
        self._trigram_commit()
        try:
            with self._db_lock:
                self.repos_db.commit()
        except Exception:
            logger.debug("Failed to commit doc_trigrams update for repo %s", repo_id, exc_info=True)

        end = perf_counter()
        logger.info(
            "_build_trigram_index: wrote_trigrams=%d db_time=%.2fs total_time=%.2fs",
            len(grams),
            end - mid,
            end - start,
        )
        # Post-commit sanity check: ensure persisted trigram keys are visible for rocksdict.
        try:
            trigram_count = self._trigram_count()
            if trigram_count == 0:
                logger.error(
                    "_build_trigram_index: post-commit trigram store reports 0 keys (backend=%s). Possible persistence failure.",
                    self._trigram_backend,
                )
            else:
                logger.info(
                    "_build_trigram_index: post-commit persisted_trigram_keys=%d (backend=%s)",
                    trigram_count,
                    self._trigram_backend,
                )
        except Exception:
            logger.debug(
                "_build_trigram_index: failed to perform post-commit trigram sanity check",
                exc_info=True,
            )
        return len(trigram_map)

    def _remove_trigrams_for_doc_ids(self, doc_ids: set[int]) -> None:
        """
        Remove trigram postings that reference the provided document IDs.

        This lets us rebuild a single repo without wiping trigram data for
        every other repo.
        """
        if not doc_ids:
            return
        fallback_scan: set[int] = set()
        for doc_id in doc_ids:
            trigrams = self._load_doc_trigrams(doc_id)
            if trigrams:
                self._remove_doc_from_trigram_postings(doc_id, trigrams)
            else:
                fallback_scan.add(doc_id)
            self._delete_doc_trigram_record(doc_id)

        # If we have no cached trigram set, fall back to a full scan for that doc_id
        for missing_doc_id in fallback_scan:
            self._remove_trigrams_for_doc_scan(missing_doc_id)

        self._trigram_commit()

    def _load_doc_trigrams(self, doc_id: int) -> set[str]:
        """Return cached trigrams for a doc_id from doc_trigrams table."""
        try:
            if hasattr(self, "metadata"):
                with self._db_lock:
                    blob = self.metadata.get_doc_trigrams_blob(doc_id)
                if not blob:
                    return set()
                return self._deserialize_trigram_set(blob)
        except Exception:
            logger.debug("Failed to load cached trigrams for doc %s via MetadataStore", doc_id, exc_info=True)

        try:
            with self._db_lock:
                cur = self.repos_db.execute(
                    "SELECT trigrams FROM doc_trigrams WHERE doc_id = ?", (doc_id,)
                )
                row = cur.fetchone()
        except Exception:
            logger.debug("Failed to load cached trigrams for doc %s", doc_id, exc_info=True)
            return set()

        if not row or row[0] is None:
            return set()
        return self._deserialize_trigram_set(row[0])

    def _store_doc_trigrams(self, doc_id: int, trigrams: set[str]) -> None:
        """Persist a trigram set for quick diff/removal later."""
        try:
            payload = self._serialize_trigram_set(trigrams)
            if hasattr(self, "metadata"):
                with self._db_lock:
                    self.metadata.store_doc_trigrams_blob(doc_id, payload)
                return
            with self._db_lock:
                self.repos_db.execute(
                    "INSERT OR REPLACE INTO doc_trigrams (doc_id, trigrams) VALUES (?, ?)",
                    (doc_id, payload),
                )
        except Exception:
            logger.debug("Failed to persist trigrams for doc %s", doc_id, exc_info=True)

    def _delete_doc_trigram_record(self, doc_id: int) -> None:
        """Remove cached trigram record for a doc_id."""
        try:
            if hasattr(self, "metadata"):
                with self._db_lock:
                    self.metadata.delete_doc_trigram_record(doc_id)
                return
            with self._db_lock:
                self.repos_db.execute("DELETE FROM doc_trigrams WHERE doc_id = ?", (doc_id,))
        except Exception:
            logger.debug("Failed to delete cached trigrams for doc %s", doc_id, exc_info=True)

    def _add_doc_to_trigram_postings(self, doc_id: int, trigrams: set[str]) -> None:
        """Add doc_id to trigram postings for provided trigrams."""
        if not trigrams:
            return
        with self._db_lock:
            for trigram in trigrams:
                existing_ids = self._trigram_get_doc_ids(trigram)
                existing_ids.add(doc_id)
                self._trigram_set_doc_ids(trigram, existing_ids)
            self._trigram_commit()

    def _remove_doc_from_trigram_postings(self, doc_id: int, trigrams: set[str]) -> None:
        """Remove doc_id from the provided trigrams' postings."""
        if not trigrams:
            return
        with self._db_lock:
            for trigram in trigrams:
                existing_ids = self._trigram_get_doc_ids(trigram)
                if not existing_ids or doc_id not in existing_ids:
                    continue
                existing_ids.discard(doc_id)
                if existing_ids:
                    self._trigram_set_doc_ids(trigram, existing_ids)
                else:
                    self._trigram_delete(trigram)
            self._trigram_commit()

    # ------------------------------------------------------------------
    # Trigram storage helpers (RocksDB backend via TrigramIndex wrapper)
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize_doc_ids(doc_ids: set[int]) -> bytes:
        """
        Store doc_ids as fixed-width little-endian uint32 array with a version prefix.
        Prefix 0x02 indicates fixed-width encoding. Older (prefix-less) and varint
        (0x01) blobs remain readable for backward compatibility.
        """
        # Delegate encoding to TrigramIndex for consistency
        try:
            return TrigramIndex._serialize_doc_ids(doc_ids)
        except Exception:
            return zlib.compress(b"")

    @staticmethod
    def _deserialize_doc_ids(blob: bytes) -> set[int]:
        # Delegate decoding to TrigramIndex for consistency
        try:
            return TrigramIndex._deserialize_doc_ids(blob)
        except Exception:
            logger.debug("Failed to deserialize doc_ids via TrigramIndex", exc_info=True)
            return set()

    def _serialize_trigram_set(self, trigrams: set[str]) -> bytes:
        # Delegate to TrigramIndex instance
        try:
            return self._trigram_index.serialize_trigram_set(trigrams)
        except Exception:
            logger.debug("Failed to serialize trigram set", exc_info=True)
            return zlib.compress(b"")

    def _deserialize_trigram_set(self, blob: bytes) -> set[str]:
        try:
            return self._trigram_index.deserialize_trigram_set(blob)
        except Exception:
            logger.debug("Failed to deserialize trigram set", exc_info=True)
            return set()

    def _trigram_commit(self) -> None:
        try:
            self._trigram_index.commit()
        except Exception:
            logger.debug("Trigram commit failed", exc_info=True)

    def _trigram_get_doc_ids(self, gram: str) -> set[int]:
        try:
            return self._trigram_index.get_doc_ids(gram)
        except Exception:
            return set()

    def _trigram_set_doc_ids(self, gram: str, doc_ids: set[int]) -> None:
        try:
            self._trigram_index.set_doc_ids(gram, doc_ids)
        except Exception:
            logger.debug("Failed to set doc_ids for trigram %s", gram, exc_info=True)
        return

    def _trigram_delete(self, gram: str) -> None:
        try:
            self._trigram_index.delete(gram)
        except Exception:
            logger.debug("Failed to delete trigram %s", gram, exc_info=True)
        return

    def _trigram_iter_items(self):
        try:
            return self._trigram_index.iter_items()
        except Exception:
            logger.debug("Failed to iterate trigram items", exc_info=True)
            return []

    def _trigram_count(self) -> int:
        try:
            return self._trigram_index.count()
        except Exception:
            logger.debug("Failed to get trigram count", exc_info=True)
            return 0

    def _extract_trigrams(self, text: str) -> set[str]:
        """Extract all trigrams from text."""
        if len(text) < 3:
            return set()
        # Fast set comprehension avoids Python loop overhead
        return {text[i : i + 3] for i in range(len(text) - 2)}

    def _read_blob(self, blob_sha: str) -> bytes | None:
        """Read blob content from storage via BlobStore."""
        try:
            if hasattr(self, "blob_store") and self.blob_store is not None:
                return self.blob_store.read_blob(blob_sha)
            blob_file = self.index_path / "blobs" / blob_sha[:2] / blob_sha[2:]
            if blob_file.exists():
                return zlib.decompress(blob_file.read_bytes())
        except Exception:
            logger.debug("Failed to read blob %s", blob_sha, exc_info=True)
        return None

    def _should_skip(self, path: Path, repo_root: Path | None = None) -> bool:
        """Check if file should be skipped during indexing."""
        # Delegate to unified should_ignore helper to maintain identical
        # semantics between watcher and indexer. Provide config/global
        # patterns and any per-repo overrides if available.
        cfg = getattr(self, "config", None)
        repo_specific = None
        if cfg is not None and repo_root is not None:
            try:
                for _rname, rraw in cfg.get("repositories", {}).items():
                    try:
                        rpath = rraw["path"] if isinstance(rraw, dict) else rraw
                    except Exception:
                        rpath = rraw
                    try:
                        rpath_s = str(rpath)
                        if Path(rpath_s).resolve() == Path(repo_root).resolve():
                            repo_specific = rraw if isinstance(rraw, dict) else None
                            break
                    except Exception:
                        continue
            except Exception:
                repo_specific = None

        repo_ignore_patterns = []
        if isinstance(repo_specific, dict):
            repo_ignore_patterns = list(repo_specific.get("ignore_patterns", []) or [])

        try:
            return should_ignore(
                path,
                repo_root,
                config_ignore_patterns=(cfg.index_ignore_patterns if cfg is not None else None),
                repo_ignore_patterns=repo_ignore_patterns,
            )
        except Exception:
            # Fail-open to legacy heuristics if helper errors
            pass
        skip_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            "target",
            "build",
            "dist",
            ".venv",
            "venv",
            ".tox",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".ruff",
            ".cache",
            "coverage",
        }
        skip_dirs.update({"htmlcov", "coverage_html", ".coverage", "site-packages"})

        skip_extensions = {
            ".pyc",
            ".so",
            ".o",
            ".a",
            ".dylib",
            ".dll",
            ".exe",
            ".bin",
            ".pdf",
            ".png",
            ".jpg",
            ".gif",
            ".svg",
            ".ico",
            ".woff",
            ".woff2",
            ".ttf",
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".xz",
            ".mjs",
        }
        skip_extensions.update({".html", ".htm", ".rmeta", ".rlib"})

        # Check if any parent is in skip_dirs
        for parent in path.parents:
            if parent.name in skip_dirs:
                return True

        # Skip common cargo build cache outputs (e.g., output/cargo_target_cache/...)
        try:
            s = str(path).lower()
            if "cargo_target_cache" in s or "/cargo_target_cache/" in s:
                return True
        except Exception:
            pass

        # Skip timestamped or backup-like files (e.g., *.backup*, *.bak, *~)
        try:
            name_l = path.name.lower()
            if ".backup" in name_l or name_l.endswith("~") or name_l.endswith(".bak"):
                return True
        except Exception:
            pass

        # Check extension
        if path.suffix.lower() in skip_extensions:
            return True

        # Skip files starting with .
        if path.name.startswith("."):
            return True

        # Skip Vite temporary build files (e.g., vite.config.ts.timestamp-*.mjs)
        if ".timestamp-" in path.name:
            return True

        # Skip large files (> 1MB)
        try:
            if path.stat().st_size > 1_000_000:
                return True
        except OSError:
            return True

        # Honor per-repo .gitignore if provided
        try:
            if repo_root:
                # If repo defines explicit includes, respect them (they override gitignore)
                include_patterns = load_include_patterns(repo_root)
                if include_patterns and is_ignored_by_gitignore(path, repo_root, include_patterns):
                    return False

                patterns = load_gitignore(repo_root)
                if patterns and is_ignored_by_gitignore(path, repo_root, patterns):
                    return True
        except Exception:
            # Fail-open: if gitignore parsing fails, do not block indexing
            pass

        return False

    def search_code(
        self, query: str, repo: str | None = None, max_results: int = 50
    ) -> list[SearchResult]:
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
            doc_ids = self._trigram_get_doc_ids(gram)
            if doc_ids:
                logger.debug("search_code: trigram %s -> doc_ids=%s", gram, sorted(doc_ids))
                doc_id_sets.append((gram, doc_ids))
            else:
                logger.debug("search_code: trigram %s not found", gram)
                # Trigram not found, no results
                return []

        # Intersect smallest postings first for efficiency; tie-break on gram for determinism
        doc_id_sets.sort(key=lambda item: (len(item[1]), item[0]))
        candidate_doc_ids = set(doc_id_sets[0][1])
        for _, posting in doc_id_sets[1:]:
            candidate_doc_ids.intersection_update(posting)
            if not candidate_doc_ids:
                return []

        # Filter by repo if specified
        if repo:
            cursor = self.repos_db.execute("SELECT id FROM repos WHERE name = ?", (repo,))
            row = cursor.fetchone()
            if row:
                repo_id = row[0]
                stmt = "SELECT id FROM documents WHERE repo_id = ? AND id IN ({})".format(
                    ",".join("?" * len(candidate_doc_ids))
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
                content = self._read_blob(doc["blob_sha"])
                if content:
                    text = content.decode("utf-8", errors="replace")
                    # Find matching lines
                    for line_num, line in enumerate(text.splitlines(), start=1):
                        if query.lower() in line.lower():
                            results.append(
                                SearchResult(
                                    repo=doc["repo_name"],
                                    path=doc["path"],
                                    line=line_num,
                                    text=line.strip(),
                                    doc_id=f"{doc['repo_name']}::{doc['path']}",
                                )
                            )
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
        self, symbol_name: str, kind: str | None = None, repo: str | None = None
    ) -> list[Symbol]:
        """
        Find symbol definitions (IDE-like "Go to Definition").

        Args:
            symbol_name: Name of symbol to find
            kind: Optional symbol kind filter (function, class, etc.)
            repo: Optional repo name to restrict search

        Returns:
            List of symbol definitions
        """
        with self._db_lock:
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
                symbols.append(
                    Symbol(
                        name=row[0],
                        kind=row[1],
                        line=row[2],
                        signature=row[3],
                        scope=row[4],
                        file_path=f"{row[6]}::{row[5]}",  # repo::path
                    )
                )

            return symbols

    def list_symbols(
        self, repo: str, file_path: str | None = None, kind: str | None = None
    ) -> list[Symbol]:
        """
        List symbols in a file or repository (IDE-like "Outline" view).

        Args:
            repo: Repository name
            file_path: Optional file path to restrict to
            kind: Optional symbol kind filter

        Returns:
            List of symbols
        """
        with self._db_lock:
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
                symbols.append(
                    Symbol(
                        name=row[0],
                        kind=row[1],
                        line=row[2],
                        signature=row[3],
                        scope=row[4],
                        file_path=row[5],
                    )
                )

            return symbols

    def _get_document(self, doc_id: int) -> dict[str, str] | None:
        """Delegate document metadata retrieval to MetadataStore."""
        try:
            if hasattr(self, "metadata"):
                with self._db_lock:
                    return self.metadata.get_document(doc_id)
        except Exception:
            logger.debug("Failed to get document %s via MetadataStore", doc_id, exc_info=True)

        # Fallback to direct SQL access
        with self._db_lock:
            cursor = self.repos_db.execute(
                """
                SELECT d.path, d.blob_sha, d.language, r.name as repo_name
                FROM documents d
                JOIN repos r ON d.repo_id = r.id
                WHERE d.id = ?
            """,
                (doc_id,),
            )
            row = cursor.fetchone()
        if row:
            return {"path": row[0], "blob_sha": row[1], "language": row[2], "repo_name": row[3]}
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
        with self._db_lock:
            # Use SymbolStore delete helper when available
            if getattr(self, "symbol_store", None) is not None:
                try:
                    self.symbol_store.delete_symbols_for_doc(doc_id)
                except Exception:
                    logger.exception("Failed to delete symbols for doc %s via SymbolStore", doc_id)
            else:
                self.repos_db.execute("DELETE FROM symbols WHERE doc_id = ?", (doc_id,))
            try:
                self.repos_db.execute("DELETE FROM embeddings WHERE doc_id = ?", (doc_id,))
            except Exception:
                # Embeddings table may not exist in newer versions; ignore errors
                logger.debug("No embeddings table found when deleting doc %s", doc_id)
            # Drop cached trigram record; postings are removed separately
            self._delete_doc_trigram_record(doc_id)

        # Attempt to delete vectors from the per-repo LanceDB if available
        if repo_id is not None:
            repo_name = self._get_repo_name_for_id(repo_id)
            if repo_name:
                try:
                    repo_db, repo_table = self._get_repo_lance_and_vectors(repo_name)
                    if repo_table is not None:
                        clauses = [f"doc_id == '{doc_id}'"]
                        if rel_path is not None:
                            clauses.append(f"file_path == '{rel_path}'")
                        for clause in clauses:
                            try:
                                self.vector_index.delete_rows(repo_table, clause)
                            except Exception:
                                logger.exception(
                                    "Failed to delete vector rows for doc %s in repo %s using clause %s",
                                    doc_id,
                                    repo_name,
                                    clause,
                                )
                        return
                except Exception:
                    logger.exception(
                        "Error deleting vectors for doc %s in repo_id %s", doc_id, repo_id
                    )

        # Fallback: try global table if present
        if self.vectors is not None:
            delete_clauses = [f"doc_id == '{doc_id}'"]
            if repo_id is not None and rel_path is not None:
                delete_clauses.append(f"repo_id == '{repo_id}' AND file_path == '{rel_path}'")

            for clause in delete_clauses:
                try:
                    self.vector_index.delete_rows(self.vectors, clause)
                except Exception:
                    logger.exception(
                        "Failed to delete vector rows for doc %s using clause %s",
                        doc_id,
                        clause,
                    )

    def _remove_trigrams_for_doc_fast(self, doc_id: int, trigrams: set[str]) -> None:
        """Fast path for removing doc_id from a list of trigrams."""
        self._remove_doc_from_trigram_postings(doc_id, trigrams)

    def _remove_trigrams_for_doc_scan(self, doc_id: int) -> None:
        """
        Slow fallback scanning removal: scan all trigrams and remove this doc_id
        where present.
        """
        with self._db_lock:
            for gram, existing_ids in self._trigram_iter_items():
                if doc_id not in existing_ids:
                    continue

                existing_ids.remove(doc_id)
                if not existing_ids:
                    self._trigram_delete(gram)
                else:
                    self._trigram_set_doc_ids(gram, existing_ids)
            self._trigram_commit()

    def _delete_blob_if_unreferenced(self, blob_sha: str, rel_path: str) -> None:
        """Delete blob file if no other document references it (delegates to BlobStore)."""
        try:
            if hasattr(self, "blob_store") and self.blob_store is not None:
                self.blob_store.delete_blob_if_unreferenced(blob_sha, rel_path)
                return
            with self._db_lock:
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
        except Exception:
            logger.debug("Failed to delete blob %s", blob_sha, exc_info=True)

    def _chunk_text(
        self, text: str, max_lines: int = 100, overlap: int = 10
    ) -> list[tuple[int, int, int, str]]:
        """Compatibility wrapper around analysis.chunk_text.

        Keeps the `indexer._chunk_text` API used throughout the codebase and tests,
        delegating the actual logic to `sigil_mcp.analysis.chunking.chunk_text`.
        """
        return chunk_text(text, max_lines=max_lines, overlap=overlap)

    def _update_vector_metadata_for_doc(
        self,
        doc_id: int,
        repo_id: int,
        rel_path: str,
    ) -> None:
        """Keep vector rows in sync when a blob moves to a different repo/path."""

        # Update metadata in the appropriate per-repo vector table if available
        repo_name = self._get_repo_name_for_id(repo_id)
        if repo_name:
            try:
                repo_db, repo_table = self._get_repo_lance_and_vectors(repo_name)
                if repo_table is not None:
                    try:
                        # Delegate update to VectorIndex helper
                        self.vector_index.update_rows(
                            repo_table,
                            where=f"doc_id == '{doc_id}'",
                            values={
                                "repo_id": str(repo_id),
                                "file_path": rel_path,
                            },
                        )
                        return
                    except Exception:
                        logger.exception(
                            "Failed to update vector metadata for doc %s in repo %s (path %s)",
                            doc_id,
                            repo_name,
                            rel_path,
                        )
            except Exception:
                logger.exception("Error looking up lance DB for repo id %s", repo_id)

        # Fallback: update global table if present
        if self.vectors is None:
            return

        try:
            self.vector_index.update_rows(
                self.vectors,
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

        # Resolve repo name and per-repo vectors table
        repo_name = self._get_repo_name_for_id(repo_id)
        repo_db = None
        vectors = None
        if repo_name:
            try:
                repo_db, vectors = self._get_repo_lance_and_vectors(repo_name)
            except Exception:
                logger.exception("Failed to get per-repo lance table for repo_id %s", repo_id)

        # If no per-repo vectors table, fall back to global one
        if vectors is None:
            vectors = self.vectors
            repo_db = self.lance_db

        if vectors is None:
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
                # Recreate table in the repo-specific DB if possible
                if repo_db is not None:
                    new_table = self.vector_index.recreate_table_for_repo(
                        repo_db, self.vector_table_name, self._code_chunk_model
                    )
                    vectors = new_table
                    if repo_name:
                        self._repo_vectors[repo_name] = new_table
                else:
                    if hasattr(self, "vector_index"):
                        try:
                            self.vectors = self.vector_index.create_table(
                                self.vector_table_name, schema=self._code_chunk_model, mode="overwrite"
                            )
                        except Exception:
                            # Fall back to direct lance DB create if available
                            self.vectors = cast(Any, self.lance_db).create_table(
                                self.vector_table_name,
                                schema=self._code_chunk_model,
                                mode="overwrite",
                            )
                    else:
                        self.vectors = cast(Any, self.lance_db).create_table(
                            self.vector_table_name,
                            schema=self._code_chunk_model,
                            mode="overwrite",
                        )
                    vectors = self.vectors
            except Exception:
                logger.exception(
                    "Failed to recreate vector table with dimension %s for %s",
                    new_dim,
                    rel_path,
                )
                return
                return

        # Normalize embeddings to unit vectors so cosine/dot metrics behave
        embeddings = embeddings.astype("float32", copy=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        timestamp = datetime.now()

        try:
            # Delete prior vectors for this file in the repo-specific table
            cast(Any, vectors).delete(f"file_path == '{rel_path}'")
        except Exception as exc:
            # If the underlying Lance dataset is missing/corrupt, recreate table and continue
            if "Not found" in str(exc) or "LanceError" in str(exc):
                logger.warning(
                    "Vector table delete failed for %s (not found/corrupt); recreating table",
                    rel_path,
                )
                try:
                    if repo_db is not None:
                        vectors = cast(Any, repo_db).create_table(
                            self.vector_table_name,
                            schema=self._code_chunk_model
                            or get_code_chunk_model(self.embedding_dimension),
                            mode="overwrite",
                        )
                        if repo_name:
                            self._repo_vectors[repo_name] = vectors
                    else:
                        if hasattr(self, "vector_index"):
                            try:
                                self.vectors = self.vector_index.create_table(
                                    self.vector_table_name,
                                    schema=self._code_chunk_model
                                    or get_code_chunk_model(self.embedding_dimension),
                                    mode="overwrite",
                                )
                            except Exception:
                                # fallback to direct lance db
                                self.vectors = cast(Any, self.lance_db).create_table(
                                    self.vector_table_name,
                                    schema=self._code_chunk_model
                                    or get_code_chunk_model(self.embedding_dimension),
                                    mode="overwrite",
                                )
                        else:
                            self.vectors = cast(Any, self.lance_db).create_table(
                                self.vector_table_name,
                                schema=self._code_chunk_model
                                or get_code_chunk_model(self.embedding_dimension),
                                mode="overwrite",
                            )
                        vectors = self.vectors
                    # After recreating the table, proceed to insert fresh records
                except Exception:
                    logger.exception("Failed to recreate vector table after delete error")
                    return
            else:
                logger.exception("Failed to delete existing vectors for %s", rel_path)
                return

        records = []
        first_chunk_text = chunks[0][3] if chunks else None
        classify = self._classify_path(rel_path, sample_text=first_chunk_text)
        for (chunk_idx, start_line, end_line, chunk_content), vector in zip(
            chunks, embeddings, strict=False
        ):
            records.append(
                {
                    "vector": np.asarray(vector, dtype="float32"),
                    "doc_id": str(doc_id),
                    "repo_id": str(repo_id),
                    "file_path": rel_path,
                    "chunk_index": int(chunk_idx),
                    "start_line": int(start_line),
                    "end_line": int(end_line),
                    "content": chunk_content,
                    "is_code": bool(classify["is_code"]),
                    "is_doc": bool(classify["is_doc"]),
                    "is_config": bool(classify["is_config"]),
                    "is_data": bool(classify["is_data"]),
                    "extension": classify["extension"],
                    "language": classify["language"],
                    "last_updated": timestamp,
                }
            )

        if records:
            try:
                # Delegate add to VectorIndex
                self.vector_index.add_records(vectors, records)
                # Mark document vectors as indexed successfully
                try:
                    with self._db_lock:
                        self.repos_db.execute(
                            "UPDATE documents SET vector_indexed_at = ?, vector_index_error = NULL WHERE id = ?",
                            (timestamp.isoformat(), str(doc_id)),
                        )
                        self.repos_db.commit()
                except Exception:
                    logger.debug(
                        "Failed to update vector_indexed_at for doc %s", doc_id, exc_info=True
                    )
            except Exception:
                logger.exception("Failed to upsert vectors for %s", rel_path)
                try:
                    with self._db_lock:
                        self.repos_db.execute(
                            "UPDATE documents SET vector_index_error = ? WHERE id = ?",
                            ("vector_upsert_failed", str(doc_id)),
                        )
                        self.repos_db.commit()
                except Exception:
                    logger.debug(
                        "Failed to record vector_index_error for doc %s", doc_id, exc_info=True
                    )

    def build_vector_index(
        self,
        repo: str,
        embed_fn: EmbeddingFn | None = None,
        model: str | None = None,
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

        with self._db_lock:
            cur = self.repos_db.cursor()
            cur.execute("SELECT id FROM repos WHERE name = ?", (repo,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Repository {repo!r} not indexed yet")
            repo_id = row[0]
            cur.execute("SELECT id, blob_sha, path FROM documents WHERE repo_id = ?", (repo_id,))
            docs = cur.fetchall()

        if self.vectors is None:
            logger.info("Vector index not initialized; skipping build for %s", repo)
            return stats

        if force and self.vectors is not None:
            try:
                self.vector_index.delete_rows(self.vectors, f"repo_id == '{repo_id}'")
            except Exception:
                logger.exception("Failed to clear existing vectors for repo %s", repo)

        for doc_id, blob_sha, rel_path in docs:
            content = self._read_blob(blob_sha)
            if not content:
                continue

            text = content.decode("utf-8", errors="replace")
            # Special-case JSONL files: produce per-record chunks
            if self._is_jsonl_path(rel_path):
                include_sol = self._get_repo_include_solution(repo_id)
                records = self._parse_jsonl_records(text, include_solution=include_sol)
                chunks = [(i, 1, 1, r) for i, r in enumerate(records)] if records else []
            else:
                chunks = self._chunk_text(text)
            # Ensure no chunk exceeds configured hard char limit
            chunks = self._enforce_chunk_size_limits(chunks)
            if not chunks:
                continue

            texts = [c[3] for c in chunks]
            if embed_fn is self.embed_fn:
                vectors = self._call_embed(texts)
            else:
                if self.embedding_provider == "llamacpp":
                    with self._embed_lock:
                        vectors = embed_fn(texts)
                else:
                    vectors = embed_fn(texts)

            self._index_file_vectors(repo_id, doc_id, rel_path, chunks, vectors)

            stats["documents_processed"] += 1
            stats["chunks_indexed"] += len(chunks)

        logger.info(
            "Built vector index for %s using model %s: %s documents, %s chunks",
            repo,
            model,
            stats["documents_processed"],
            stats["chunks_indexed"],
        )
        self._log_vector_index_status(context=f"rebuild:{repo}")
        return stats

    def generate_hardwrap_report(self, repo: str | None = None, top_n: int = 50) -> list[dict]:
        """Scan documents and report files that trigger many hard-wrap splits.

        Returns a list of dicts: {repo: name, path: rel_path, oversized_count: int, total_chunks: int}
        sorted by oversized_count descending.
        """
        results: list[dict] = []
        with self._db_lock:
            cur = self.repos_db.cursor()
            if repo:
                repo_names = [repo]
            else:
                cur.execute("SELECT name FROM repos")
                repo_names = [r[0] for r in cur.fetchall()]

        for repo_name in repo_names:
            with self._db_lock:
                cur = self.repos_db.cursor()
                cur.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
                row = cur.fetchone()
                if not row:
                    continue
                repo_id = row[0]
                cur.execute(
                    "SELECT id, blob_sha, path FROM documents WHERE repo_id = ?", (repo_id,)
                )
                doc_rows = cur.fetchall()

            for _doc_id, blob_sha, rel_path in doc_rows:
                try:
                    content = self._read_blob(blob_sha)
                    if not content:
                        continue
                    text = content.decode("utf-8", errors="replace")
                    if self._is_jsonl_path(rel_path):
                        include_sol = self._get_repo_include_solution(repo_id)
                        records = self._parse_jsonl_records(text, include_solution=include_sol)
                        chunks = [(i, 1, 1, r) for i, r in enumerate(records)] if records else []
                    else:
                        chunks = self._chunk_text(text)
                    cfg = get_config()
                    hard_chars = cfg.embed_hard_chars
                    max_tokens = cfg.embeddings_max_tokens
                    oversized = 0
                    for c in chunks:
                        txt = c[3]
                        if not txt:
                            continue
                        if len(txt) > hard_chars:
                            oversized += 1
                            continue
                        toks = self._count_tokens(txt)
                        if toks > max_tokens:
                            oversized += 1
                    if oversized:
                        results.append(
                            {
                                "repo": repo_name,
                                "path": rel_path,
                                "oversized_count": oversized,
                                "total_chunks": len(chunks),
                            }
                        )
                except Exception:
                    continue
        results.sort(key=lambda x: x.get("oversized_count", 0), reverse=True)
        return results[:top_n]

    def semantic_search(
        self,
        query: str,
        repo: str | None = None,
        k: int = 20,
        embed_fn: EmbeddingFn | None = None,
        model: str | None = None,
        code_only: bool = False,
        prefer_code: bool = False,
        candidate_limit: int | None = None,
    ) -> list[dict[str, object]]:
        """
        Semantic code search using vector embeddings.

        Args:
            query: Natural language or code query
            repo: Repository name to search (optional; searches all if omitted)
            k: Number of top results to return
            embed_fn: Embedding function (uses instance default if None)
            model: Model identifier (uses instance default if None)
            code_only: If True, hard filter to code chunks (is_code == True)
            prefer_code: If True, rerank to favor code but still allow docs/config
            candidate_limit: Optional override for the number of vector hits to fetch before reranking

        Returns:
            List of search results with scores, sorted by relevance
        """
        if embed_fn is None:
            embed_fn = self.embed_fn
        if embed_fn is None or self.vectors is None:
            logger.info(
                "Semantic search requested but embeddings are unavailable; " "returning no results."
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

        # Fetch a broader candidate pool for reranking
        rerank_pool = candidate_limit or max(k * 5, 50)
        rerank_pool = min(max(rerank_pool, k), 200)
        if code_only:
            rerank_pool = max(rerank_pool, k * 5)
        elif prefer_code:
            rerank_pool = max(rerank_pool, k * 4)

        # 1) embed query outside DB locks
        q_vec = self._call_embed([query])[0].astype("float32")
        q_norm = np.linalg.norm(q_vec) or 1.0
        q_vec = q_vec / q_norm

        def _lance_search(table: object, vec: np.ndarray, limit: int) -> list[dict]:
            """Search a LanceDB table preferring cosine metric when available."""
            try:
                return (
                    cast(Any, table)
                    .search(vec.astype("float32"))
                    .metric("cosine")
                    .limit(limit)
                    .to_list()
                )
            except Exception:
                return (
                    cast(Any, table).search(vec.astype("float32")).limit(limit).to_list()
                )

        def _rows_to_candidates(rows: list[dict], repo_name: str | None) -> list[dict]:
            out: list[dict] = []
            for r in rows:
                try:
                    doc_id = int(r.get("doc_id", 0))
                except (TypeError, ValueError):
                    continue
                if repo_name is None:
                    doc = self._get_document(doc_id)
                    repo_n = doc.get("repo_name") if doc else None
                else:
                    repo_n = repo_name
                if repo_n is None:
                    continue
                distance = r.get("_distance")
                if distance is None:
                    try:
                        vec = np.asarray(r.get("vector"), dtype="float32")
                        if vec.shape == q_vec.shape:
                            distance = float(1.0 - float(np.dot(vec, q_vec)))
                        else:
                            distance = 1.0
                    except Exception:
                        distance = 1.0

                candidate = {
                    "repo": repo_n,
                    "path": r.get("file_path", ""),
                    "chunk_index": int(r.get("chunk_index", -1)),
                    "start_line": int(r.get("start_line", 0)),
                    "end_line": int(r.get("end_line", 0)),
                    "content": r.get("content", ""),
                    "_distance": float(distance),
                    "is_code": bool(r.get("is_code", True)),
                    "is_doc": bool(r.get("is_doc", False)),
                    "is_config": bool(r.get("is_config", False)),
                    "is_data": bool(r.get("is_data", False)),
                    "extension": r.get("extension"),
                    "language": r.get("language"),
                    "doc_id": f"{repo_n}::{r.get('file_path', '')}",
                }
                out.append(candidate)
            return out

        def _rerank(candidates: list[dict]) -> list[dict]:
            reranked: list[dict] = []
            for c in candidates:
                is_code = bool(c.get("is_code", True))
                is_doc = bool(c.get("is_doc", False))
                is_config = bool(c.get("is_config", False))
                is_data = bool(c.get("is_data", False))
                if code_only and not is_code:
                    continue
                base_distance = max(float(c.get("_distance", 1.0)), 1e-6)
                score = 1.0 / (1.0 + base_distance)

                penalty = 1.0
                if prefer_code:
                    if is_code:
                        penalty *= 1.2
                    elif is_doc:
                        penalty *= 0.8
                    elif is_config:
                        penalty *= 0.85
                    elif is_data:
                        penalty *= 0.9
                else:
                    if is_doc:
                        penalty *= 0.85
                    elif is_config:
                        penalty *= 0.9
                    elif is_data:
                        penalty *= 0.9
                    elif is_code:
                        penalty *= 1.05

                final_score = score * penalty
                # Prevent runaway >1.0 scores from boosting multipliers
                final_score = min(final_score, 1.0)

                result = dict(c)
                result["score"] = final_score
                # Expose whether language is authoritative (only for code)
                result["language_authoritative"] = bool(is_code)
                reranked.append(result)

            reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return reranked[:k]

        # If a repo was specified, search only its per-repo table
        if repo:
            with self._db_lock:
                cur = self.repos_db.cursor()
                cur.execute("SELECT id FROM repos WHERE name = ?", (repo,))
                row = cur.fetchone()
            if not row:
                raise ValueError(f"Repository {repo!r} not indexed yet")
            # Use per-repo vectors if available
            repo_db, repo_table = self._get_repo_lance_and_vectors(repo)
            if repo_table is None:
                return []
            try:
                query_results = _lance_search(repo_table, q_vec, rerank_pool)
            except Exception:
                logger.exception("Semantic search failed for repo %s", repo)
                return []
            return _rerank(_rows_to_candidates(query_results, repo))

        # No repo specified: aggregate from all per-repo tables if present
        aggregated: list[dict] = []
        try:
            with self._db_lock:
                cur = self.repos_db.cursor()
                cur.execute("SELECT name FROM repos")
                repo_names = [r[0] for r in cur.fetchall()]
            for rname in repo_names:
                try:
                    repo_db, repo_table = self._get_repo_lance_and_vectors(rname)
                    if repo_table is None:
                        continue
                    rows = _lance_search(repo_table, q_vec, rerank_pool)
                    for res in _rows_to_candidates(rows, rname):
                        aggregated.append(res)
                except Exception:
                    logger.debug("Failed semantic search for repo %s", rname, exc_info=True)
            # Sort aggregated results by score and return top-k
            return _rerank(aggregated)
        except Exception:
            logger.exception("Semantic search aggregation failed")
            return []

    def get_index_stats(self, repo: str | None = None) -> dict[str, int | str]:
        """Get statistics about the index."""
        with self._db_lock:
            cursor = self.repos_db.cursor()

            if repo:
                cursor.execute("SELECT id FROM repos WHERE name = ?", (repo,))
                row = cursor.fetchone()
                if not row:
                    return {"error": "Repository not found"}
                repo_id = row[0]

                cursor.execute("SELECT COUNT(*) FROM documents WHERE repo_id = ?", (repo_id,))
                doc_count = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM symbols WHERE doc_id IN "
                    "(SELECT id FROM documents WHERE repo_id = ?)",
                    (repo_id,),
                )
                symbol_count = cursor.fetchone()[0]

                cursor.execute("SELECT indexed_at FROM repos WHERE id = ?", (repo_id,))
                indexed_at = cursor.fetchone()[0]

                return {
                    "repo": repo,
                    "documents": doc_count,
                    "symbols": symbol_count,
                    "indexed_at": indexed_at,
                }
            else:
                cursor.execute("SELECT COUNT(*) FROM repos")
                repo_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM documents")
                doc_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM symbols")
                symbol_count = cursor.fetchone()[0]

                # Query trigrams from the trigrams database
                trigram_count = self._trigram_count()

                return {
                    "repositories": repo_count,
                    "documents": doc_count,
                    "symbols": symbol_count,
                    "trigrams": trigram_count,
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
        try:
            with self._db_lock:
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
            trigrams = self._load_doc_trigrams(doc_id)
            if not trigrams:
                content = self._read_blob(blob_sha)
                if content is not None:
                    text = content.decode("utf-8", errors="replace").lower()
                    trigrams = self._extract_trigrams(text)
                    logger.warning(
                        "remove_file: found %d trigrams for doc %s",
                        len(trigrams),
                        doc_id,
                    )

            # Delete symbols and embeddings for this doc and clear vectors
            self._delete_symbols_and_embeddings_for_doc(doc_id, repo_id, rel_path)

            # Update trigram index to drop this doc_id
            if trigrams:
                self._remove_doc_from_trigram_postings(doc_id, trigrams)
            else:
                # Fallback scan to remove doc_id from any trigram postings found
                self._remove_trigrams_for_doc_scan(doc_id)

            # Delete document row and cached trigram record
            with self._db_lock:
                self.repos_db.execute(
                    "DELETE FROM documents WHERE id = ?",
                    (doc_id,),
                )
                self._delete_doc_trigram_record(doc_id)
                self.repos_db.commit()

            # Optionally delete blob content if no other docs reference it
            self._delete_blob_if_unreferenced(blob_sha, rel_path)

            self._trigram_commit()
            logger.warning(
                "remove_file: committed deletion for doc_id %s " "and updated trigrams",
                doc_id,
            )

            logger.info("Removed %s from index (repo=%s)", rel_path, repo_name)
            return True
        except Exception as exc:
            logger.error("Error removing %s from index: %s", file_path, exc)
            return False

    def close(self) -> None:
        """Close all database connections."""
        # Close trigram index (backend-agnostic)
        if hasattr(self, "_trigram_index") and getattr(self, "_trigram_index", None) is not None:
            try:
                self._trigram_index.close()
            except Exception:
                logger.debug("Error closing trigram index", exc_info=True)

        # Close SQLite repos database
        if hasattr(self, "repos_db") and self.repos_db is not None:
            try:
                self.repos_db.close()
            except Exception:
                logger.debug("Error closing repos database", exc_info=True)
