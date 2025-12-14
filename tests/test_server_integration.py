# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import json
import types
from io import BytesIO

import pytest
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

import sigil_mcp.server as server
from sigil_mcp.config import Config
from sigil_mcp.indexer import SigilIndex


@pytest.fixture
def live_server(monkeypatch, tmp_path, test_repo_path, dummy_embed_fn):
    """Create a real index and config, and wire server globals."""
    index_path = tmp_path / "idx"
    index_path.mkdir(parents=True, exist_ok=True)
    idx = SigilIndex(index_path, embed_fn=dummy_embed_fn, embed_model="test-model")
    idx.index_repository("test_repo", test_repo_path, force=True)

    cfg_data = {
        "mode": "dev",
        "admin": {"enabled": True, "require_api_key": False, "allowed_ips": ["127.0.0.1"]},
        "authentication": {"enabled": False, "allow_local_bypass": True, "oauth_enabled": False},
        "index": {"path": str(index_path)},
        "repositories": {"test_repo": str(test_repo_path)},
        "embeddings": {"enabled": True, "dimension": 768},
        "server": {"log_file": str(tmp_path / "server.log")},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg_data))
    cfg = Config(cfg_path)

    monkeypatch.setattr(server, "config", cfg)
    monkeypatch.setattr(server, "REPOS", {"test_repo": test_repo_path})
    monkeypatch.setattr(
        server,
        "REPO_OPTIONS",
        {"test_repo": {"path": test_repo_path, "respect_gitignore": True, "ignore_patterns": []}},
    )
    monkeypatch.setattr(server, "_INDEX", idx)
    monkeypatch.setattr(server, "_WATCHER", None)
    try:
        yield {"index": idx, "config": cfg, "repo_path": test_repo_path}
    finally:
        idx.close()


def test_repo_path_resolution_and_list_files(live_server):
    repo_path = live_server["repo_path"]
    assert server._get_repo_root("test_repo") == repo_path
    resolved = server._resolve_under_repo("test_repo", "main.py")
    assert resolved.exists()
    with pytest.raises(ValueError):
        server._resolve_under_repo("test_repo", "../outside")

    listing = server.list_repo_files("test_repo", max_depth=2)
    assert listing["entries"]
    assert any(e["path"].endswith("main.py") for e in listing["entries"])


def test_search_and_fetch_roundtrip(live_server):
    results = server.search_repo("hello_world", repo="test_repo", file_glob="*.py", max_results=5)
    assert results
    fetch = server.fetch("test_repo::main.py")
    assert "Hello" in fetch["text"]


def test_index_and_vector_stats(live_server):
    stats = server.get_index_stats("test_repo")
    assert stats.get("documents", 0) >= 1

    vec = server.build_vector_index("test_repo", force_rebuild=True, model="default")
    assert vec.get("chunks_indexed", 0) >= 0
    assert vec["model"] == "default"


def test_search_and_symbol_wrappers(live_server):
    results = server.search("hello_world", repo="test_repo", file_glob="*.py", max_results=5)
    assert results["results"]

    code_results = server.search_code("hello_world", repo="test_repo", max_results=5)
    assert any(r["doc_id"].startswith("test_repo::") for r in code_results)

    definitions = server.goto_definition("hello_world", repo="test_repo")
    assert definitions and definitions[0]["name"] == "hello_world"

    symbols = server.list_symbols("test_repo")
    assert symbols

    # Wrapper reindexes without forcing rebuild
    stats = server.index_repository("test_repo", force_rebuild=False)
    assert stats["status"] == "completed"


def test_list_repo_files_truncated(live_server):
    listing = server.list_repo_files("test_repo", max_entries=1)
    assert listing["truncated"] is True
    assert len(listing["entries"]) <= 1


def test_read_repo_file_truncates(live_server):
    repo_path = live_server["repo_path"]
    big_file = repo_path / "big.txt"
    big_file.write_text("a" * 5000)
    content = server.read_repo_file("test_repo", "big.txt", max_bytes=1000)
    assert "[... truncated ...]" in content


def test_semantic_search_without_embeddings(live_server):
    index = live_server["index"]
    index.embed_fn = None
    index.vectors = None
    resp = server.semantic_search("hello", repo="test_repo")
    assert resp["status"] == "error"
    assert "Semantic search is not available" in resp["error"]


def test_build_vector_index_op_skip_branches(monkeypatch, tmp_path):
    # embeddings disabled
    monkeypatch.setattr(server, "REPOS", {"repo": tmp_path})
    monkeypatch.setattr(server, "_get_index", lambda: types.SimpleNamespace())
    monkeypatch.setattr(server, "get_config", lambda: types.SimpleNamespace(embeddings_enabled=False))
    disabled = server.build_vector_index_op(repo="repo", force_rebuild=False)
    assert disabled["reason"] == "embeddings_disabled"

    # lancedb unavailable
    monkeypatch.setattr(server, "_get_index", lambda: types.SimpleNamespace(lancedb_available=False))
    monkeypatch.setattr(server, "get_config", lambda: types.SimpleNamespace(embeddings_enabled=True))
    missing = server.build_vector_index_op(repo="repo", force_rebuild=False)
    assert missing["reason"] == "lancedb_missing"


def test_external_mcp_status_and_refresh_error(monkeypatch):
    monkeypatch.setattr(server, "get_global_manager", lambda: None)
    status = server.external_mcp_status_op()
    assert status["enabled"] is False
    with pytest.raises(RuntimeError):
        server.refresh_external_mcp_op()


def test_mcp_bearer_auth_middleware_branches(monkeypatch):
    app = Starlette()

    async def home(request):
        return PlainTextResponse("ok")

    app.add_route("/", home, methods=["GET"])

    app.add_middleware(
        server.MCPBearerAuthMiddleware,
        token="token123",
        require_token=True,
        allow_local_bypass=False,
    )
    client = TestClient(app)
    assert client.get("/").status_code == 401
    assert client.get("/", headers={"Authorization": "Bearer token123"}).status_code == 200

    # Local bypass branch
    app2 = Starlette()

    async def home2(request):
        return PlainTextResponse("ok")

    app2.add_route("/", home2, methods=["GET"])

    monkeypatch.setattr(server, "is_local_connection", lambda ip=None: True)
    app2.add_middleware(
        server.MCPBearerAuthMiddleware,
        token="secret",
        require_token=True,
        allow_local_bypass=True,
    )
    client2 = TestClient(app2)
    assert client2.get("/").status_code == 200


def test_get_form_value_handles_uploadfile():
    upload = UploadFile(filename="tmp.txt", file=BytesIO(b"data"))
    assert server.get_form_value("plain") == "plain"
    assert server.get_form_value(upload) is None
    assert server.get_form_value(None) is None


def test_search_wrapper_and_fetch_error(live_server):
    with pytest.raises(ValueError):
        server.fetch("invalid-doc-id")

    # valid fetch still works
    result = server.search_repo("hello_world", repo="test_repo", file_glob="*.py", max_results=1)[0]
    doc_id = f"{result['repo']}::{result['path']}"
    fetched = server.fetch(doc_id)
    assert fetched["id"] == doc_id


def test_rebuild_and_vector_ops_use_live_index(live_server, monkeypatch):
    cfg = live_server["config"]
    idx = live_server["index"]
    monkeypatch.setattr(server, "get_config", lambda: cfg)
    monkeypatch.setattr(server, "_INDEX", idx)
    monkeypatch.setattr(server, "_get_index", lambda: idx)
    monkeypatch.setattr(server, "REPOS", {"test_repo": live_server["repo_path"]})
    monkeypatch.setattr(server, "_ensure_repos_configured", lambda: None)

    rebuild = server.rebuild_index_op(repo="test_repo", force_rebuild=False)
    assert "files_indexed" in rebuild

    vec = server.build_vector_index_op(repo="test_repo", force_rebuild=False, model="default")
    assert vec["status"] in {"completed", "skipped"}
