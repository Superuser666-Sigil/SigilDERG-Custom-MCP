import os
from sigil_mcp.storage.metadata import MetadataStore


def test_metadata_store_init(tmp_path):
    db_path = tmp_path / "repos.db"
    ms = MetadataStore(db_path)
    assert db_path.exists()
    rid = ms.ensure_repo("testrepo", str(tmp_path))
    assert isinstance(rid, int)
    # ensure get_repo_id works
    got = ms.get_repo_id("testrepo")
    assert got == rid
    ms.close()
