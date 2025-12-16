"""Vector index wrapper around LanceDB (or in-memory stub).

Provides thin helpers used by the indexer (create/open table, add, delete, search).
This module intentionally keeps a minimal surface so it can be wired into the
refactor in small steps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, cast

logger = logging.getLogger(__name__)

try:
    import lancedb  # type: ignore
    LANCEDB_AVAILABLE = True
except Exception:
    lancedb: Any = None
    LANCEDB_AVAILABLE = False


class VectorIndex:
    def __init__(self, base_path: Path, dimension: int, use_stub: bool = False):
        self.base_path = base_path
        self.dimension = dimension
        self._db: Any = None
        self._tables: dict[str, Any] = {}
        self.use_stub = use_stub or (not LANCEDB_AVAILABLE)
        self._init_db()

        # Registry for per-repo DB stubs when using stub mode
        self._repo_registry: dict[str, Any] = {}

    def _init_db(self) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)
        if self.use_stub:
            # Lightweight in-memory stub
            self._db = _InMemoryStub(self.dimension)
        else:
            try:
                self._db = cast(Any, lancedb).connect(str(self.base_path))
            except Exception:
                logger.exception("Failed to initialize LanceDB; falling back to stub")
                self._db = _InMemoryStub(self.dimension)

    def table(self, name: str) -> Optional[Any]:
        try:
            if name in self._tables:
                return self._tables[name]
            if self._db is None:
                return None
            if hasattr(self._db, "table_names") and name in set(self._db.table_names()):
                tbl = self._db.open_table(name)
                self._tables[name] = tbl
                return tbl
            return None
        except Exception:
            logger.debug("Failed to open vector table %s", name, exc_info=True)
            return None

    def get_repo_db(self, repo_path: Path):
        """Return a per-repo LanceDB connection or stub for the given path."""
        pkey = str(repo_path)
        if self.use_stub:
            if pkey in self._repo_registry:
                return self._repo_registry[pkey]
            stub = _InMemoryStub(self.dimension)
            self._repo_registry[pkey] = stub
            return stub

        try:
            return cast(Any, lancedb).connect(pkey) if lancedb else None
        except Exception:
            logger.exception("Failed to connect to LanceDB at %s", pkey)
            return None

    def create_table(self, name: str, schema: Any, mode: str = "create") -> Any:
        # Delegate to LanceDB or stub
        if self._db is None:
            raise RuntimeError("Vector DB unavailable")
        if hasattr(self._db, "create_table"):
            return self._db.create_table(name, schema=schema, mode=mode)
        return self._db.create_table(name, schema=schema, mode=mode)

    # Higher-level helpers used by indexer to operate on repo-specific DBs/tables
    def create_table_for_repo(self, repo_db: Any, name: str, schema: Any, mode: str = "create") -> Any:
        """Create (or overwrite) a table in a per-repo LanceDB instance or stub.

        Falls back to the global base DB if repo_db is None or does not support
        create_table.
        """
        try:
            if repo_db is not None and hasattr(repo_db, "create_table"):
                return repo_db.create_table(name, schema=schema, mode=mode)
        except Exception:
            logger.exception("Failed to create table on repo DB; falling back to base DB")

        # Fallback to base DB
        return self.create_table(name, schema=schema, mode=mode)

    def recreate_table_for_repo(self, repo_db: Any, name: str, schema: Any) -> Any:
        """Force overwrite/recreate a table for a repo.

        Returns the newly created table instance.
        """
        return self.create_table_for_repo(repo_db, name, schema, mode="overwrite")

    def add_records(self, table: Any, records: list[dict]) -> None:
        try:
            if table is None:
                raise RuntimeError("No target table to add records")
            if hasattr(table, "add"):
                table.add(records)
                return
            # Some LanceDB objects may wrap add under different names; attempt generic call
            getattr(table, "add")(records)
        except Exception:
            logger.exception("Failed to add records to vector table")

    def delete_rows(self, table: Any, where: str) -> None:
        try:
            if table is None:
                return
            if hasattr(table, "delete"):
                table.delete(where)
                return
            getattr(table, "delete")(where)
        except Exception:
            logger.exception("Failed to delete rows from vector table (where=%s)", where)

    def update_rows(self, table: Any, where: str, values: dict) -> None:
        try:
            if table is None:
                return
            if hasattr(table, "update"):
                table.update(where=where, values=values)
                return
            getattr(table, "update")(where=where, values=values)
        except Exception:
            logger.exception("Failed to update rows on vector table (where=%s)", where)

    def count_rows(self, table: Any) -> int:
        try:
            if table is None:
                return 0
            if hasattr(table, "count_rows"):
                return int(table.count_rows())
            if hasattr(table, "count"):
                return int(table.count())
            return 0
        except Exception:
            logger.exception("Failed to count rows on vector table")
            return 0

    def search_table(self, table: Any, vector: Any, metric: str | None = None, limit: int | None = None):
        try:
            q = table.search(vector)
            if metric and hasattr(q, "metric"):
                q = q.metric(metric)
            if limit and hasattr(q, "limit"):
                q = q.limit(limit)
            return q
        except Exception:
            logger.exception("Vector search failed")
            return None


# Minimal in-memory stub used when lancedb is not available
class _InMemoryStub:
    def __init__(self, dim: int):
        self.dim = dim
        self._tables: dict[str, list] = {}

    def table_names(self):
        return list(self._tables.keys())

    def open_table(self, name: str):
        if name not in self._tables:
            raise KeyError("Table not found")
        return _InMemoryTable(self._tables[name])

    def create_table(self, name: str, schema: Any = None, mode: str = "create"):
        if mode == "overwrite" or name not in self._tables:
            self._tables[name] = []
        return _InMemoryTable(self._tables[name])


class _InMemoryTable:
    def __init__(self, rows: list):
        self._rows = rows

    def add(self, records: list[dict]):
        # Replace existing rows for same (repo_id, doc_id, file_path, chunk_index)
        for r in records:
            self._rows.append(r)

    def delete(self, where: str):
        # best-effort: support simple equality checks like "file_path == 'x'" or "repo_id == '1'"
        if "==" not in where:
            return
        lhs, rhs = where.split("==", 1)
        lhs = lhs.strip()
        rhs = rhs.strip().strip("'\"")
        new = [r for r in self._rows if str(r.get(lhs)) != rhs]
        self._rows[:] = new

    def update(self, where: str, values: dict):
        # naive implementation used only in tests/refactor
        if "==" not in where:
            return
        lhs, rhs = where.split("==", 1)
        lhs = lhs.strip()
        rhs = rhs.strip().strip("'\"")
        for r in self._rows:
            if str(r.get(lhs)) == rhs:
                r.update(values)

    def count_rows(self) -> int:
        return len(self._rows)

    def to_arrow(self):
        """Return a minimal Arrow-like object with to_pylist() for tests."""
        class _Arrow:
            def __init__(self, rows):
                self._rows = list(rows)

            def to_pylist(self):
                return list(self._rows)

        return _Arrow(self._rows)

    def search(self, vec: Any):
        # Return an object with `metric`, `limit`, `to_list` used by indexer
        class Q:
            def __init__(self, rows):
                self._rows = rows

            def metric(self, _m):
                return self

            def limit(self, n):
                self._n = n
                return self

            def to_list(self):
                # Return up to n arbitrary rows (no scoring) in fallback
                return self._rows[: getattr(self, "_n", len(self._rows))]

        return Q(self._rows)


