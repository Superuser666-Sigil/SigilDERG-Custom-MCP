# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import sqlite3
from pathlib import Path

from sigil_mcp.scripts import rebuild_indexes as script


class DummyIndex:
    def __init__(self, tmp_path: Path):
        self.trigrams_db = sqlite3.connect(tmp_path / "tri.db")
        self.trigrams_db.execute("CREATE TABLE IF NOT EXISTS trigrams (gram TEXT PRIMARY KEY, doc_ids BLOB)")
        self.trigrams_db.execute("INSERT INTO trigrams VALUES ('abc', '0')")
        self.repos_db = sqlite3.connect(tmp_path / "repos.db")
        self.embed_fn = None
        self.embed_model = "none"

    def index_repository(self, repo_name, repo_path, force=True):
        return {"files_indexed": 1, "trigrams_built": 1}

    def build_vector_index(self, repo, embed_fn, model, force=True):
        return {"documents_processed": 1, "chunks_indexed": 1}


def test_delete_all_trigrams(tmp_path):
    idx = DummyIndex(tmp_path)
    deleted = script.delete_all_trigrams(idx)
    assert deleted == 1


def test_rebuild_trigrams_and_embeddings(tmp_path):
    idx = DummyIndex(tmp_path)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    trig_stats = script.rebuild_trigrams_for_repo(idx, "r1", repo_path)
    assert trig_stats["trigrams_built"] == 1
    emb_stats = script.rebuild_embeddings_for_repo(idx, "r1", lambda x: x, "model")
    assert emb_stats["documents_processed"] == 1


def test_rebuild_all_indexes_no_embeddings(monkeypatch, tmp_path):
    repos = {"r1": str(tmp_path / "repo")}
    Path(repos["r1"]).mkdir(parents=True, exist_ok=True)

    class DummyConfig:
        repositories = repos
        embeddings_enabled = False
        index_path = tmp_path
        lance_dir = tmp_path / "lancedb"

    monkeypatch.setattr(script, "get_config", lambda: DummyConfig)
    idx = DummyIndex(tmp_path)
    result = script.rebuild_all_indexes(index=idx, wipe_index=False, rebuild_embeddings=False)
    assert result["success"] is True
    assert "trigram_stats" in result


def test_rebuild_all_indexes_with_embeddings(monkeypatch, tmp_path):
    repos = {"r1": str(tmp_path / "repo")}
    Path(repos["r1"]).mkdir(parents=True, exist_ok=True)

    class DummyProvider:
        def embed_documents(self, texts):
            import numpy as np
            return [list(np.zeros(2)) for _ in texts]

        def get_dimension(self):
            return 2

    class DummyConfig:
        repositories = repos
        embeddings_enabled = True
        embeddings_provider = "dummy"
        embeddings_model = "m"
        embeddings_kwargs = {}
        embeddings_cache_dir = None
        embeddings_api_key = None
        embeddings_dimension = 2
        index_path = tmp_path
        lance_dir = tmp_path / "lancedb"

    monkeypatch.setattr(script, "get_config", lambda: DummyConfig)
    monkeypatch.setattr(script, "create_embedding_provider", lambda **kwargs: DummyProvider())
    idx = DummyIndex(tmp_path)
    res = script.rebuild_all_indexes(index=idx, wipe_index=False, rebuild_embeddings=True)
    assert res["embedding_stats"]
