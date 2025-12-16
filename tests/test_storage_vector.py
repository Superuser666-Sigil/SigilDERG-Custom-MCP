from sigil_mcp.storage.vector import VectorIndex


def test_vector_index_stub(tmp_path):
    v = VectorIndex(tmp_path / "lance", 128, use_stub=True)
    tbl = v.create_table("code_vectors", schema=None)
    # add rows
    tbl.add([{"repo_id": "1", "doc_id": "1", "file_path": "a", "chunk_index": 0}])
    assert tbl.count_rows() == 1
    # delete rows by file_path
    tbl.delete("file_path == 'a'")
    assert tbl.count_rows() == 0


def test_vector_index_per_repo(tmp_path):
    v = VectorIndex(tmp_path / "lance", 64, use_stub=True)
    repo_path = tmp_path / "repo1"
    repo_path.mkdir()
    repo_db = v.get_repo_db(repo_path)
    tbl = repo_db.create_table("code_vectors", schema=None)
    tbl.add([{"repo_id": "1", "doc_id": "2", "file_path": "b", "chunk_index": 0}])
    assert tbl.count_rows() == 1
    # opening same repo returns same stub instance
    repo_db2 = v.get_repo_db(repo_path)
    assert repo_db is repo_db2
