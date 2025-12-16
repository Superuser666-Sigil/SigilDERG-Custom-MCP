from sigil_mcp.storage.vector import VectorIndex


def test_vector_create_add_count_delete_update_search(tmp_path):
    vi = VectorIndex(base_path=tmp_path, dimension=4, use_stub=True)

    repo_id = 'repo1'
    repo_db = vi.get_repo_db(tmp_path / repo_id)
    tbl = vi.create_table_for_repo(repo_db, 'table1', schema=None)

    # Initially empty
    assert vi.count_rows(tbl) == 0

    # Add some records
    records = [
        {'id': 'r1', 'vector': [0.1, 0.2, 0.3, 0.4], 'metadata': {'path': 'a.py'}},
        {'id': 'r2', 'vector': [1.0, 0.0, 0.0, 0.0], 'metadata': {'path': 'b.py'}},
    ]
    vi.add_records(tbl, records)
    assert vi.count_rows(tbl) == 2

    # Search for nearest neighbor to vector similar to r1
    q = vi.search_table(tbl, [0.1, 0.2, 0.3, 0.4], limit=1)
    res = q.to_list()
    assert len(res) == 1

    # Update a row (naive stub update)
    vi.update_rows(tbl, "id == 'r1'", {'vector': [0.0, 0.0, 0.0, 0.0]})
    # delete r2
    vi.delete_rows(tbl, "id == 'r2'")
    assert vi.count_rows(tbl) == 1

    # Recreate the table (overwrite)
    vi.recreate_table_for_repo(repo_db, 'table1', schema=None)
    new_tbl = vi.create_table_for_repo(repo_db, 'table1', schema=None)
    assert vi.count_rows(new_tbl) == 0
