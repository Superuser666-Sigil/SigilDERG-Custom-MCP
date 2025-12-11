# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import pytest
import sigil_mcp.config as sigil_config
from sigil_mcp.indexer import SigilIndex


@pytest.fixture
def lancedb_indexed_repo(embeddings_enabled_index):
    ctx = embeddings_enabled_index
    index = ctx["index"]
    repo_path = ctx["repo_path"]
    repo_name = ctx["repo_name"]

    index.index_repository(repo_name, repo_path, force=True)
    return ctx


def test_build_vector_index_writes_to_lancedb(lancedb_indexed_repo):
    index = lancedb_indexed_repo["index"]
    repo_name = lancedb_indexed_repo["repo_name"]

    stats = index.build_vector_index(repo_name, force=True)

    assert stats["chunks_indexed"] > 0
    assert index.vectors is not None
    assert index.vectors.count_rows() > 0


def test_semantic_search_uses_lancedb(lancedb_indexed_repo, monkeypatch):
    index = lancedb_indexed_repo["index"]
    repo_name = lancedb_indexed_repo["repo_name"]

    index.build_vector_index(repo_name, force=True)

    search_calls = []
    original_search = index.vectors.search

    def spy_search(vector):
        search_calls.append(vector)
        return original_search(vector)

    monkeypatch.setattr(index.vectors, "search", spy_search)

    results = index.semantic_search("calculator", repo=repo_name, k=3)

    assert search_calls, "LanceDB search should be invoked"
    assert isinstance(results, list)


def test_remove_file_clears_lancedb_rows(lancedb_indexed_repo):
    index = lancedb_indexed_repo["index"]
    repo_path = lancedb_indexed_repo["repo_path"]
    repo_name = lancedb_indexed_repo["repo_name"]

    index.build_vector_index(repo_name, force=True)

    rel_path = "main.py"
    before_rows = [
        row
        for row in index.vectors.to_arrow().to_pylist()
        if row.get("file_path") == rel_path
    ]
    assert before_rows

    removed = index.remove_file(repo_name, repo_path, repo_path / rel_path)

    after_rows = [
        row
        for row in index.vectors.to_arrow().to_pylist()
        if row.get("file_path") == rel_path
    ]
    assert removed is True
    assert not after_rows


def test_embeddings_disabled_paths_raise_errors(temp_dir, test_repo_path):
    original_config = sigil_config._config
    cfg = sigil_config.Config()
    cfg.config_data.setdefault("embeddings", {})["enabled"] = False
    cfg.config_data.setdefault("index", {})["path"] = str(temp_dir / ".disabled_vectors")
    sigil_config._config = cfg

    index = SigilIndex(index_path=cfg.index_path)
    index.index_repository("test_repo", test_repo_path, force=True)

    try:
        stats = index.build_vector_index("test_repo")
        assert stats["documents_processed"] == 0

        results = index.semantic_search("test", repo="test_repo")
        assert results == []
    finally:
        index.repos_db.close()
        index.trigrams_db.close()
        sigil_config._config = original_config
