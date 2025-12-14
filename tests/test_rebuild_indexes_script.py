# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import json
from pathlib import Path

import numpy as np
import pytest

from sigil_mcp.indexer import SigilIndex
from sigil_mcp.scripts import rebuild_indexes as script
import sigil_mcp.config as sigil_config


@pytest.fixture
def live_index(monkeypatch, tmp_path, test_repo_path, dummy_embed_fn):
    """Create a real SigilIndex against a temp repo and temp config."""
    index_path = tmp_path / "idx"
    index_path.mkdir(parents=True, exist_ok=True)

    cfg = sigil_config.Config()
    cfg.config_data.update(
        {
            "mode": "dev",
            "embeddings": {"enabled": True, "dimension": 768},
            "index": {"path": str(index_path), "ignore_patterns": []},
            "repositories": {"test_repo": str(test_repo_path)},
            "server": {"log_file": str(tmp_path / "server.log")},
        }
    )
    monkeypatch.setattr(sigil_config, "_config", cfg)

    idx = SigilIndex(index_path, embed_fn=dummy_embed_fn, embed_model="test-model")
    idx.index_repository("test_repo", test_repo_path, force=True)

    try:
        yield {"index": idx, "config": cfg, "repo_path": test_repo_path}
    finally:
        idx.close()


def test_delete_all_trigrams(live_index):
    idx = live_index["index"]
    # Ensure trigrams exist
    initial = idx._trigram_count()
    assert initial > 0

    deleted = script.delete_all_trigrams(idx)
    assert deleted == initial
    assert idx._trigram_count() == 0


def test_rebuild_trigrams_and_embeddings(live_index):
    idx = live_index["index"]
    repo_path = live_index["repo_path"]

    trig_stats = script.rebuild_trigrams_for_repo(idx, "test_repo", repo_path)
    assert trig_stats["trigrams_built"] >= 1

    emb_stats = script.rebuild_embeddings_for_repo(idx, "test_repo", idx.embed_fn, idx.embed_model)
    assert emb_stats["documents_processed"] >= 1


def test_rebuild_all_indexes_no_embeddings(monkeypatch, tmp_path, test_repo_path):
    # Build config on disk so load_config can read it
    cfg_path = tmp_path / "config.json"
    cfg_data = {
        "mode": "dev",
        "embeddings": {"enabled": False},
        "index": {"path": str(tmp_path / "idx")},
        "repositories": {"r1": str(test_repo_path)},
        "server": {"log_file": str(tmp_path / "server.log")},
    }
    cfg_path.write_text(json.dumps(cfg_data))

    idx_path = tmp_path / "idx"
    idx_path.mkdir(parents=True, exist_ok=True)

    idx = SigilIndex(idx_path, embed_fn=None, embed_model="none")
    idx.index_repository("r1", test_repo_path, force=True)

    class DummyConfig:
        def __init__(self, data):
            self.config_data = data
            self.repositories = data["repositories"]
            self.embeddings_enabled = False
            self.index_path = Path(data["index"]["path"])
            self.lance_dir = self.index_path / "lancedb"

    monkeypatch.setattr(script, "_find_config_path", lambda: cfg_path)
    monkeypatch.setattr(script, "load_config", lambda path: DummyConfig(cfg_data))
    res = script.rebuild_all_indexes(index=idx, wipe_index=False, rebuild_embeddings=False)
    assert res["success"] is True
    assert "trigram_stats" in res
    idx.repos_db.close()


def test_rebuild_all_indexes_with_embeddings(monkeypatch, tmp_path, test_repo_path):
    cfg_path = tmp_path / "config.json"
    cfg_data = {
        "mode": "dev",
        "embeddings": {
            "enabled": True,
            "provider": "dummy",
            "model": "m",
            "dimension": 2,
            "kwargs": {},
        },
        "index": {"path": str(tmp_path / "idx")},
        "repositories": {"r1": str(test_repo_path)},
        "server": {"log_file": str(tmp_path / "server.log")},
    }
    cfg_path.write_text(json.dumps(cfg_data))

    idx = SigilIndex(tmp_path / "idx", embed_fn=lambda texts: np.zeros((len(texts), 2), dtype="float32"))
    idx.index_repository("r1", test_repo_path, force=True)

    class DummyProvider:
        def embed_documents(self, texts):
            import numpy as np
            return [list(np.zeros(2)) for _ in texts]

        def get_dimension(self):
            return 2

    class DummyConfig:
        def __init__(self, data):
            self.config_data = data
            self.repositories = data["repositories"]
            self.embeddings_enabled = True
            self.embeddings_provider = data["embeddings"]["provider"]
            self.embeddings_model = data["embeddings"]["model"]
            self.embeddings_kwargs = data["embeddings"].get("kwargs", {})
            self.embeddings_cache_dir = None
            self.embeddings_api_key = None
            self.embeddings_dimension = data["embeddings"]["dimension"]
            self.index_path = Path(data["index"]["path"])
            self.lance_dir = self.index_path / "lancedb"

    monkeypatch.setattr(script, "_find_config_path", lambda: cfg_path)
    monkeypatch.setattr(script, "load_config", lambda path: DummyConfig(cfg_data))
    monkeypatch.setattr(script, "create_embedding_provider", lambda **kwargs: DummyProvider())
    res = script.rebuild_all_indexes(index=idx, wipe_index=False, rebuild_embeddings=True)
    assert res["embedding_stats"]
    idx.repos_db.close()


def test_find_config_path_walks_parents(monkeypatch, tmp_path):
    root = tmp_path / "root"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    cfg = root / "config.json"
    cfg.write_text("{}")
    monkeypatch.chdir(nested)
    found = script._find_config_path()
    assert found == cfg


def test_setup_index_for_rebuild_wipes_directories(monkeypatch, tmp_path):
    index_dir = tmp_path / "idx"
    lance_dir = tmp_path / "lance"
    index_dir.mkdir()
    lance_dir.mkdir()
    (index_dir / "old.db").write_text("x")
    (lance_dir / "old.db").write_text("y")

    class DummyConfig:
        def __init__(self):
            self.index_path = index_dir
            self.lance_dir = lance_dir

    created = {}

    class DummyIndex:
        def __init__(self, path, embed_fn=None, embed_model=None):
            created["path"] = path
            created["embed_fn"] = embed_fn
            created["embed_model"] = embed_model

    monkeypatch.setattr(script, "_find_config_path", lambda: None)
    monkeypatch.setattr(script, "get_config", lambda: DummyConfig())
    monkeypatch.setattr(script, "SigilIndex", DummyIndex)

    res = script._setup_index_for_rebuild(index=None, wipe_index=True)
    assert isinstance(res, DummyIndex)
    # Directories recreated but previous files should be gone
    assert (index_dir / "old.db").exists() is False
    assert (lance_dir / "old.db").exists() is False


def test_rebuild_all_indexes_raises_when_no_repos(monkeypatch, tmp_path):
    idx_path = tmp_path / "idx"
    lance_path = tmp_path / "lance"
    class DummyConfig:
        def __init__(self):
            self.repositories = {}
            self.index_path = idx_path
            self.lance_dir = lance_path
            self.embeddings_enabled = False

    dummy_index = object()
    monkeypatch.setattr(script, "_find_config_path", lambda: None)
    monkeypatch.setattr(script, "get_config", lambda: DummyConfig())
    with pytest.raises(ValueError):
        script.rebuild_all_indexes(index=dummy_index, wipe_index=False)


def test_setup_embedding_function_requires_provider(monkeypatch):
    class DummyConfig:
        embeddings_provider = None
        embeddings_model = None
        embeddings_kwargs = {}
        embeddings_cache_dir = None
        embeddings_api_key = None
        embeddings_dimension = 2

    with pytest.raises(ValueError):
        script._setup_embedding_function(DummyConfig())


def test_rebuild_single_repo_index_validations(tmp_path):
    class DummyIndex:
        def index_repository(self, repo, repo_path, force=True):
            return {"files_indexed": 0, "symbols_extracted": 0, "trigrams_built": 0}

    dummy_index = DummyIndex()
    missing_repo = tmp_path / "missing"
    with pytest.raises(ValueError):
        script.rebuild_single_repo_index(dummy_index, "missing", missing_repo)

    existing_repo = tmp_path / "repo"
    existing_repo.mkdir()
    with pytest.raises(ValueError):
        script.rebuild_single_repo_index(dummy_index, "repo", existing_repo, rebuild_embeddings=True, embed_fn=None)


def test_main_handles_success_and_error(monkeypatch, capsys):
    called = {}

    def fake_rebuild_all_indexes(**kwargs):
        called["called"] = True
        return {
            "deleted_trigrams": 1,
            "repos": {"r1": {"files": 1, "trigrams": 2}},
            "embedding_stats": {"r1": {"documents_processed": 1, "chunks_indexed": 1}},
        }

    monkeypatch.setattr(script, "rebuild_all_indexes", fake_rebuild_all_indexes)
    assert script.main() == 0
    out = capsys.readouterr().out
    assert "Embedding rebuild summary" in out

    monkeypatch.setattr(script, "rebuild_all_indexes", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
    assert script.main() == 1


def test_find_config_path_searches_module_parents(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("{}")
    fake_file = tmp_path / "pkg" / "scripts" / "rebuild_indexes.py"
    fake_file.parent.mkdir(parents=True)
    monkeypatch.setattr(script, "__file__", str(fake_file))
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)
    assert script._find_config_path() == cfg_path


def test_find_config_path_uses_module_parent_when_cwd_tree_empty(monkeypatch, tmp_path_factory):
    cfg_root = tmp_path_factory.mktemp("cfgmod")
    cfg_path = cfg_root / "config.json"
    cfg_path.write_text("{}")
    fake_file = cfg_root / "pkg" / "scripts" / "rebuild_indexes.py"
    fake_file.parent.mkdir(parents=True)
    monkeypatch.setattr(script, "__file__", str(fake_file))

    cwd = tmp_path_factory.mktemp("othercwd")
    monkeypatch.chdir(cwd)

    assert script._find_config_path() == cfg_path


def test_find_config_path_returns_none_when_missing(monkeypatch, tmp_path_factory):
    fake_root = tmp_path_factory.mktemp("nofiles")
    fake_file = fake_root / "scripts" / "rebuild_indexes.py"
    fake_file.parent.mkdir(parents=True)
    monkeypatch.setattr(script, "__file__", str(fake_file))
    other_cwd = tmp_path_factory.mktemp("othercwd2")
    monkeypatch.chdir(other_cwd)

    assert script._find_config_path() is None


def test_delete_all_trigrams_handles_exception():
    class BadIndex:
        def _trigram_count(self):
            raise RuntimeError("fail")

    assert script.delete_all_trigrams(BadIndex()) == 0


def test_rebuild_trigrams_skips_missing_repo(tmp_path):
    idx = object()
    stats = script._rebuild_trigrams_for_all_repos(idx, {"missing": str(tmp_path / "missing")})
    assert stats == {}


def test_setup_embedding_function_with_cache_and_key(monkeypatch):
    class DummyProvider:
        def __init__(self):
            self.called = True

        def embed_documents(self, texts):
            return [[0.0] for _ in texts]

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return DummyProvider()

    cfg = type(
        "Cfg",
        (),
        {
            "embeddings_provider": "openai",
            "embeddings_model": "m",
            "embeddings_kwargs": {},
            "embeddings_cache_dir": "/tmp/cache",
            "embeddings_api_key": "token",
            "embeddings_dimension": 1,
        },
    )()
    monkeypatch.setattr(script, "create_embedding_provider", fake_create)
    embed_fn, model = script._setup_embedding_function(cfg)
    assert model == "openai:m"
    assert captured["cache_dir"] == "/tmp/cache"
    assert captured["api_key"] == "token"
    arr = embed_fn(["x"])
    assert arr.shape[0] == 1


def test_rebuild_single_repo_index_with_embeddings(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "file.txt").write_text("x")

    class DummyIndex:
        def index_repository(self, repo, path, force=True):
            return {"files_indexed": 1, "symbols_extracted": 0, "trigrams_built": 1}

        def build_vector_index(self, repo, embed_fn, model, force=True):
            return {"documents_processed": 1, "chunks_indexed": 1}

    def embed_fn(texts):
        import numpy as np
        return np.zeros((len(texts), 1), dtype="float32")

    res = script.rebuild_single_repo_index(
        DummyIndex(), "repo", repo_path, rebuild_embeddings=True, embed_fn=embed_fn, model="m"
    )
    assert res["embedding_stats"]["documents_processed"] == 1
