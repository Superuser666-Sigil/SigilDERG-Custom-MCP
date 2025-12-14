# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import types
from pathlib import Path
import numpy as np

import pytest

import sigil_mcp.server as server


def test_get_admin_app_returns_starlette_app():
    app = server._get_admin_app()
    # Calling twice should use cached instance
    assert app is server._get_admin_app()
    assert hasattr(app, "routes")


def test_ensure_repos_configured_raises(monkeypatch):
    monkeypatch.setattr(server, "REPOS", {})
    with pytest.raises(RuntimeError):
        server._ensure_repos_configured()


def test_create_embedding_function_missing_provider(monkeypatch):
    monkeypatch.setattr(
        server,
        "config",
        types.SimpleNamespace(
            embeddings_enabled=True,
            embeddings_provider=None,
            embeddings_model=None,
            embeddings_dimension=256,
            embeddings_kwargs={},
            embeddings_cache_dir=None,
            embeddings_api_key=None,
        ),
    )
    monkeypatch.setattr(server, "READINESS", {"config": True, "index": False, "embeddings": False})
    embed_fn, model = server._create_embedding_function()
    assert embed_fn is None and model is None


def test_create_embedding_function_import_error(monkeypatch):
    monkeypatch.setattr(
        server,
        "config",
        types.SimpleNamespace(
            embeddings_enabled=True,
            embeddings_provider="custom",
            embeddings_model="demo",
            embeddings_dimension=64,
            embeddings_kwargs={},
            embeddings_cache_dir=None,
            embeddings_api_key=None,
        ),
    )
    monkeypatch.setattr(
        "sigil_mcp.embeddings.create_embedding_provider",
        lambda **kwargs: (_ for _ in ()).throw(ImportError("missing")),
    )
    embed_fn, model = server._create_embedding_function()
    assert embed_fn is None and model is None


def test_create_embedding_function_success(monkeypatch):
    class DummyProvider:
        def embed_documents(self, texts):
            return [[float(len(t))] for t in texts]

    monkeypatch.setattr(
        "sigil_mcp.embeddings.create_embedding_provider",
        lambda **kwargs: DummyProvider(),
    )
    monkeypatch.setattr(
        server,
        "config",
        types.SimpleNamespace(
            embeddings_enabled=True,
            embeddings_provider="dummy",
            embeddings_model="model",
            embeddings_dimension=1,
            embeddings_kwargs={},
            embeddings_cache_dir=None,
            embeddings_api_key=None,
        ),
    )
    monkeypatch.setattr(server, "READINESS", {"config": True, "index": False, "embeddings": False})
    embed_fn, model = server._create_embedding_function()
    assert model == "dummy:model"
    arr = embed_fn(["abc"])
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 1)


def test_get_index_initializes_when_missing(monkeypatch, tmp_path):
    # Use a fresh index path and ensure globals are reset during the test
    monkeypatch.setattr(server, "_INDEX", None)
    index_path = tmp_path / "idx"
    index_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        server,
        "config",
        types.SimpleNamespace(
            index_path=index_path,
            embeddings_enabled=False,
            embeddings_provider=None,
            embeddings_model=None,
            embeddings_dimension=0,
            embeddings_kwargs={},
            embeddings_cache_dir=None,
            embeddings_api_key=None,
            admin_enabled=False,
            allow_local_bypass=True,
            auth_enabled=False,
            oauth_enabled=False,
            watch_enabled=False,
            watch_ignore_dirs=[],
            watch_ignore_extensions=[],
            mode="dev",
            mcp_sse_path="/sse",
            mcp_message_path="/message",
            mcp_http_path="/",
            mcp_require_token=False,
            mcp_server_token=None,
        ),
    )
    index = server._get_index()
    try:
        assert index.index_path.exists()
    finally:
        index.close()


def test_get_watcher_and_start_watching(monkeypatch, tmp_path):
    watched = []

    class DummyWatcher:
        def __init__(self, on_change, ignore_dirs=None, ignore_extensions=None):
            self.on_change = on_change
            self.ignore_dirs = ignore_dirs
            self.ignore_extensions = ignore_extensions
            self.watched = []

        def start(self):
            return None

        def watch_repository(self, name, path, honor_gitignore=True, repo_ignore_patterns=None):
            self.watched.append((name, path, honor_gitignore, repo_ignore_patterns))

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    monkeypatch.setattr(server, "FileWatchManager", DummyWatcher)
    monkeypatch.setattr(server, "_WATCHER", None)
    monkeypatch.setattr(
        server,
        "config",
        types.SimpleNamespace(
            watch_enabled=True,
            watch_ignore_dirs=[".git"],
            watch_ignore_extensions=[".pyc"],
        ),
    )
    monkeypatch.setattr(server, "REPOS", {"repo": repo_path})
    monkeypatch.setattr(server, "REPO_OPTIONS", {"repo": {"respect_gitignore": False}})
    watcher = server._get_watcher()
    assert watcher is not None
    server._start_watching_repos()
    assert watcher.watched


def test_on_file_change_deleted(monkeypatch, tmp_path):
    removed = {}

    class DummyIndex:
        def remove_file(self, repo_name, repo_path, file_path):
            removed["args"] = (repo_name, repo_path, file_path)
            return True

    dummy_index = DummyIndex()
    monkeypatch.setattr(server, "_INDEX", dummy_index)
    monkeypatch.setattr(server, "_get_index", lambda: dummy_index)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    monkeypatch.setattr(server, "REPOS", {"repo": repo_path})
    server._on_file_change("repo", repo_path / "file.py", "deleted")
    assert removed["args"][0] == "repo"


def test_setup_authentication_enabled(monkeypatch):
    class DummyOAuthManager:
        def initialize_client(self):
            return ("id", "secret")

        def get_client(self):
            return types.SimpleNamespace(client_id="id")

    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setattr(server, "OAUTH_ENABLED", True)
    monkeypatch.setattr(server, "ALLOW_LOCAL_BYPASS", True)
    monkeypatch.setattr(server, "ALLOWED_IPS", ["127.0.0.1"])
    monkeypatch.setattr(server, "initialize_api_key", lambda: "abc123")
    monkeypatch.setattr(server, "get_oauth_manager", lambda: DummyOAuthManager())

    # Should run without raising and exercise logging branches
    server._setup_authentication()


def test_setup_authentication_disabled(monkeypatch):
    monkeypatch.setattr(server, "AUTH_ENABLED", False)
    server._setup_authentication()


def test_setup_file_watching_invokes_start(monkeypatch):
    called = {}
    monkeypatch.setattr(server, "_start_watching_repos", lambda: called.setdefault("called", True))
    monkeypatch.setattr(
        server,
        "config",
        types.SimpleNamespace(watch_enabled=True, watch_debounce_seconds=1, watch_ignore_dirs=[], watch_ignore_extensions=[]),
    )
    monkeypatch.setattr(server, "REPOS", {"r": Path(".")})
    server._setup_file_watching()
    assert called.get("called") is True


def test_build_sse_app_with_token(monkeypatch):
    middleware_created = {}

    class DummyMiddleware:
        def __init__(self, cls, **kw):
            middleware_created["token"] = kw.get("token")
            middleware_created["require_token"] = kw.get("require_token")

    def fake_create_sse_app(**kwargs):
        return {"middleware": kwargs["middleware"], "sse_path": kwargs["sse_path"], "message_path": kwargs["message_path"]}

    monkeypatch.setattr(server, "Middleware", DummyMiddleware)
    monkeypatch.setattr(server, "create_sse_app", fake_create_sse_app)
    monkeypatch.setattr(
        server,
        "config",
        types.SimpleNamespace(
            mcp_require_token=True,
            mcp_server_token="t",
            mcp_sse_path="/sse",
            mcp_message_path="/msg",
        ),
    )
    dummy_mcp = types.SimpleNamespace(
        _auth_server_provider=None,
        _additional_http_routes=None,
        settings=types.SimpleNamespace(debug=False),
    )
    monkeypatch.setattr(server, "mcp", dummy_mcp)
    app = server._build_sse_app(sse_route_path="/sse", message_route_path="/msg")
    assert middleware_created["require_token"] is True
    assert middleware_created["token"] == "t"
    assert app["sse_path"] == "/sse"


def test_get_index_stats_op_repo(monkeypatch, tmp_path):
    repo_path = tmp_path / "r"
    repo_path.mkdir()
    (repo_path / "file.txt").write_text("x")
    monkeypatch.setattr(server, "REPOS", {"r": repo_path})

    class FakeVectors:
        def __init__(self, count):
            self._count = count

        def count_rows(self, filter=None):
            return self._count

    class FakeCursor:
        def __init__(self, responses):
            self._responses = list(responses)

        def execute(self, *a, **k):
            return None

        def fetchone(self):
            if self._responses:
                return self._responses.pop(0)
            return (0,)

    class FakeReposDB:
        def __init__(self, responses):
            self.responses = responses

        def cursor(self):
            return FakeCursor(self.responses)

    class FakeIndex:
        def __init__(self):
            self.repos_db = FakeReposDB([(1,), (0,)])
            self.vectors = FakeVectors(5)

        def get_index_stats(self, repo=None):
            return {"documents": 1, "symbols": 2, "repositories": 1}

    monkeypatch.setattr(server, "_get_index", lambda: FakeIndex())
    stats = server.get_index_stats_op(repo="r")
    assert stats["repos"]["r"]["vectors"] == 5


def test_get_index_stats_op_all_repos(monkeypatch, tmp_path):
    repo_path = tmp_path / "r"
    repo_path.mkdir()
    (repo_path / "file.txt").write_text("x")
    monkeypatch.setattr(server, "REPOS", {"r": repo_path})

    class FakeVectors:
        def count_rows(self, filter=None):
            return 3

    class FakeCursor:
        def __init__(self):
            self.calls = 0

        def execute(self, *a, **k):
            self.calls += 1

        def fetchone(self):
            return (1,)

    class FakeReposDB:
        def cursor(self):
            return FakeCursor()

    class FakeIndex:
        def __init__(self):
            self.repos_db = FakeReposDB()
            self.vectors = FakeVectors()

        def get_index_stats(self, repo=None):
            if repo:
                return {"documents": 1, "symbols": 1}
            return {"documents": 2, "symbols": 3, "repositories": 1}

    monkeypatch.setattr(server, "_get_index", lambda: FakeIndex())
    stats = server.get_index_stats_op(repo=None)
    assert stats["total_vectors"] == 3


def test_semantic_search_success(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "REPOS", {"r": tmp_path})

    class FakeIndex:
        def __init__(self):
            self.embed_fn = object()
            self.vectors = object()
            self.embed_model = "model"

        def semantic_search(self, **kwargs):
            return [{"repo": kwargs.get("repo"), "score": 1.0}]

    monkeypatch.setattr(server, "_get_index", lambda: FakeIndex())
    res = server.semantic_search("query", repo="r", k=1, model="default")
    assert res["status"] == "completed"
