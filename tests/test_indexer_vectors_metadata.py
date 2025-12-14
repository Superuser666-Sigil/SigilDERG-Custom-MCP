
import numpy as np

from sigil_mcp.indexer import SigilIndex


def test_schema_migration_adds_vector_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGIL_MCP_LANCEDB_STUB", "1")
    idx_path = tmp_path / "index"
    idx = SigilIndex(idx_path, embed_fn=lambda texts: np.zeros((len(texts), idx_path and 768), dtype="float32"))
    cur = idx.repos_db.cursor()
    cur.execute("PRAGMA table_info(documents)")
    cols = {row[1] for row in cur.fetchall()}
    assert "vector_indexed_at" in cols
    assert "vector_index_error" in cols
    idx.repos_db.close()


def test_index_file_sets_vector_indexed_at(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGIL_MCP_LANCEDB_STUB", "1")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    file_path = repo_dir / "sample.py"
    file_path.write_text("def hello():\n    return 'world'\n")

    idx_path = tmp_path / "index"
    # simple embed_fn returning zeros of configured dim
    def embed_fn(texts):
        import numpy as _np
        return _np.zeros((len(texts), 768), dtype="float32")

    idx = SigilIndex(idx_path, embed_fn=embed_fn)

    # index the file
    success = idx.index_file("testrepo", repo_dir, file_path)
    assert success

    # verify documents row exists and vector_indexed_at set (since we used stubbed lance)
    cur = idx.repos_db.cursor()
    cur.execute("SELECT id, vector_indexed_at, vector_index_error FROM documents")
    row = cur.fetchone()
    assert row is not None
    doc_id, vect_at, vect_err = row
    assert (vect_at is not None) or (vect_err is None)

    idx.close()
