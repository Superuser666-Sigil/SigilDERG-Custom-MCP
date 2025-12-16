"""Symbol storage helpers (SQLite-backed).

Provides a thin wrapper around the `symbols` table to encapsulate
CRUD operations so `indexer.py` can delegate symbol persistence.
"""
from __future__ import annotations

import sqlite3
from typing import Iterable, List

from ..analysis import Symbol


class SymbolStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def store_symbols_for_doc(self, doc_id: int, symbols: Iterable[Symbol]) -> None:
        """Insert the provided symbols for a document.

        This method performs an executemany insert into the `symbols`
        table. Caller should hold any necessary locks and commit the
        transaction as appropriate.
        """
        symbols = list(symbols)
        if not symbols:
            return
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO symbols (
                doc_id, name, kind, line, character, signature, scope
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    doc_id,
                    s.name,
                    s.kind,
                    s.line,
                    0,
                    s.signature,
                    s.scope,
                )
                for s in symbols
            ],
        )

    def delete_symbols_for_doc(self, doc_id: int) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM symbols WHERE doc_id = ?", (doc_id,))

    def get_symbols_for_doc(self, doc_id: int) -> List[Symbol]:
        """Return Symbol objects for a given document id by joining
        the documents table to provide file path information.
        """
        cur = self.conn.cursor()
        rows = list(
            cur.execute(
                """
                SELECT s.name, s.kind, d.path, s.line, s.signature, s.scope
                FROM symbols s
                JOIN documents d ON s.doc_id = d.id
                WHERE s.doc_id = ?
                """,
                (doc_id,),
            )
        )
        return [
            Symbol(name=r[0], kind=r[1], file_path=r[2], line=r[3], signature=r[4], scope=r[5])
            for r in rows
        ]
