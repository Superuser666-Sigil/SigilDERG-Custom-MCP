import sqlite3

from sigil_mcp.storage.symbols import SymbolStore
from sigil_mcp.analysis.symbols import Symbol


def test_symbol_store_crud(tmp_path):
    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()
    # Minimal tables used by SymbolStore.get_symbols_for_doc
    cur.execute("""
    CREATE TABLE documents (
        id INTEGER PRIMARY KEY,
        path TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE symbols (
        doc_id INTEGER,
        name TEXT,
        kind TEXT,
        line INTEGER,
        character INTEGER,
        signature TEXT,
        scope TEXT
    )
    """)
    conn.commit()

    # Insert document row
    cur.execute("INSERT INTO documents (id, path) VALUES (?, ?)", (1, 'a.py'))
    conn.commit()

    store = SymbolStore(conn)

    symbols = [
        Symbol(name='f1', kind='function', file_path='a.py', line=10, signature=None, scope=None),
        Symbol(name='C', kind='class', file_path='a.py', line=1, signature=None, scope=None),
    ]

    # Store symbols
    store.store_symbols_for_doc(1, symbols)

    # Retrieve via SQL directly
    cur.execute("SELECT COUNT(*) FROM symbols WHERE doc_id = ?", (1,))
    count = cur.fetchone()[0]
    assert count == 2

    # get_symbols_for_doc should return Symbol objects with file_path from documents
    got = store.get_symbols_for_doc(1)
    assert len(got) == 2
    names = {s.name for s in got}
    assert names == {"f1", "C"}

    # Delete and ensure gone
    store.delete_symbols_for_doc(1)
    cur.execute("SELECT COUNT(*) FROM symbols WHERE doc_id = ?", (1,))
    assert cur.fetchone()[0] == 0
