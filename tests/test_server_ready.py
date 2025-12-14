# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).

import asyncio
import types

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import sigil_mcp.server as server
from sigil_mcp.indexer import SigilIndex


def test_healthz_and_readyz_direct_call():
    resp = asyncio.run(server.healthz(None))
    assert resp.status_code == 200
    ready = asyncio.run(server.readyz(None))
    assert ready.status_code in {200, 503}
    import json
    payload = json.loads(ready.body.decode())
    assert "components" in payload


def test_oauth_http_errors(monkeypatch):
    monkeypatch.setattr(server, "OAUTH_ENABLED", True)
    dummy_mgr = types.SimpleNamespace(
        verify_client=lambda cid, secret=None: False,
        get_client=lambda: None,
    )
    monkeypatch.setattr(server, "get_oauth_manager", lambda: dummy_mgr)
    app = Starlette(routes=[Route("/oauth/authorize", server.oauth_authorize_http, methods=["GET", "POST"])])
    client = TestClient(app)

    # missing params
    resp = client.get("/oauth/authorize")
    assert resp.status_code == 400

    # invalid client
    resp2 = client.get(
        "/oauth/authorize",
        params={
            "client_id": "bad",
            "redirect_uri": "https://example.com",
            "response_type": "code",
        },
    )
    assert resp2.status_code == 401

    # invalid response type
    ok_mgr = types.SimpleNamespace(
        verify_client=lambda cid, secret=None: True,
        get_client=lambda: types.SimpleNamespace(redirect_uris=["https://good.example"]),
    )
    monkeypatch.setattr(server, "get_oauth_manager", lambda: ok_mgr)
    resp3 = client.get(
        "/oauth/authorize",
        params={
            "client_id": "good",
            "redirect_uri": "https://good.example",
            "response_type": "token",
        },
    )
    assert resp3.status_code == 400

    # invalid redirect not in allow list
    resp4 = client.get(
        "/oauth/authorize",
        params={
            "client_id": "good",
            "redirect_uri": "https://bad.example",
            "response_type": "code",
        },
    )
    assert resp4.status_code == 400


def test_build_vector_index_op_initializes_embeddings(monkeypatch, tmp_path, test_repo_path):
    index_path = tmp_path / "idx"
    index_path.mkdir()
    index = SigilIndex(index_path, embed_fn=None, embed_model="none")
    index.index_repository("test_repo", test_repo_path, force=True)
    # ensure vector flow is allowed
    index.vectors = object()
    index.lancedb_available = True
    index._embeddings_active = True

    class DummyProvider:
        def __init__(self):
            self.calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            return [[float(len(t))] for t in texts]

    provider = DummyProvider()
    monkeypatch.setattr("sigil_mcp.embeddings.create_embedding_provider", lambda **kwargs: provider)
    cfg = types.SimpleNamespace(
        embeddings_enabled=True,
        embeddings_provider="dummy",
        embeddings_model="m",
        embeddings_dimension=1,
        embeddings_kwargs={},
        embeddings_cache_dir=None,
        embeddings_api_key=None,
    )
    monkeypatch.setattr(server, "_INDEX", index)
    monkeypatch.setattr(server, "get_config", lambda: cfg)
    monkeypatch.setattr(server, "REPOS", {"test_repo": test_repo_path})
    monkeypatch.setattr(server, "_ensure_repos_configured", lambda: None)

    result = server.build_vector_index_op(repo="test_repo", force_rebuild=False, model="default")
    assert result["success"] is True
    assert provider.calls >= 1


def test_build_vector_index_op_all_repos(monkeypatch, tmp_path):
    # Dummy index that records calls
    class FakeIndex:
        def __init__(self):
            self.embed_fn = None
            self.embed_model = None
            self.lancedb_available = True
            self.calls = []

        def build_vector_index(self, repo, embed_fn, model, force):
            self.calls.append((repo, embed_fn is not None, model, force))
            return {"documents_processed": 0}

    fake_index = FakeIndex()
    monkeypatch.setattr(server, "_INDEX", fake_index)
    monkeypatch.setattr(server, "_get_index", lambda: fake_index)
    monkeypatch.setattr(server, "_ensure_repos_configured", lambda: None)
    monkeypatch.setattr(server, "REPOS", {"r1": tmp_path / "r1", "r2": tmp_path / "r2"})

    cfg = types.SimpleNamespace(
        embeddings_enabled=True,
        embeddings_provider="dummy",
        embeddings_model="m",
        embeddings_dimension=1,
        embeddings_kwargs={},
        embeddings_cache_dir=None,
        embeddings_api_key=None,
    )
    monkeypatch.setattr(server, "get_config", lambda: cfg)

    class Provider:
        def embed_documents(self, texts):
            return [[1.0] for _ in texts]

    monkeypatch.setattr("sigil_mcp.embeddings.create_embedding_provider", lambda **kwargs: Provider())
    result = server.build_vector_index_op(repo=None, force_rebuild=True, model="default")
    assert result["success"] is True
    assert len(fake_index.calls) == 2


def test_build_vector_index_op_embeddings_unavailable(monkeypatch, tmp_path):
    class FakeIndex:
        def __init__(self):
            self.embed_fn = None
            self.embed_model = None
            self.lancedb_available = True

    fake_index = FakeIndex()
    monkeypatch.setattr(server, "_INDEX", fake_index)
    monkeypatch.setattr(server, "_get_index", lambda: fake_index)
    monkeypatch.setattr(server, "_ensure_repos_configured", lambda: None)
    monkeypatch.setattr(server, "REPOS", {"r1": tmp_path / "r1"})

    cfg = types.SimpleNamespace(
        embeddings_enabled=True,
        embeddings_provider="dummy",
        embeddings_model="m",
        embeddings_dimension=1,
        embeddings_kwargs={},
        embeddings_cache_dir=None,
        embeddings_api_key=None,
    )
    monkeypatch.setattr(server, "get_config", lambda: cfg)
    monkeypatch.setattr(
        "sigil_mcp.embeddings.create_embedding_provider",
        lambda **kwargs: (_ for _ in ()).throw(ImportError("no backend")),
    )
    result = server.build_vector_index_op(repo=None, force_rebuild=False, model="default")
    assert result["status"] == "skipped"
    assert result["reason"] == "embeddings_unavailable"
