# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import sigil_mcp.admin_api as admin_api
import sigil_mcp.server as server_state
from sigil_mcp.config import Config
from sigil_mcp.indexer import SigilIndex


@pytest.fixture
def admin_app(monkeypatch, tmp_path, test_repo_path, dummy_embed_fn):
    """Provision a real SigilIndex and config against a temp repo and wire into admin_api."""
    index_path = tmp_path / "idx"
    index_path.mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / "server.log"
    log_path.write_text("line1\nline2\nline3\n")

    # Build a real index with rocksdict backend and stubbed embeddings
    idx = SigilIndex(index_path, embed_fn=dummy_embed_fn, embed_model="test-model")
    idx.index_repository("test_repo", test_repo_path, force=True)

    cfg_data = {
        "mode": "dev",
        "admin": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 9999,
            "api_key": "secret",
            "require_api_key": False,
            "allowed_ips": ["testclient", "127.0.0.1"],
        },
        "authentication": {
            "enabled": False,
            "allow_local_bypass": True,
            "oauth_enabled": False,
            "allowed_ips": ["testclient", "127.0.0.1"],
        },
        "index": {"path": str(index_path)},
        "repositories": {"test_repo": str(test_repo_path)},
        "embeddings": {"enabled": True, "dimension": 768},
        "server": {"log_file": str(log_path)},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg_data))
    cfg = Config(cfg_path)

    # Wire globals to use the temp config/index
    monkeypatch.setattr(admin_api, "_config", cfg)
    monkeypatch.setattr(server_state, "config", cfg)
    monkeypatch.setattr(server_state, "REPOS", {"test_repo": test_repo_path})
    monkeypatch.setattr(
        server_state,
        "REPO_OPTIONS",
        {"test_repo": {"path": test_repo_path, "respect_gitignore": True, "ignore_patterns": []}},
    )
    monkeypatch.setattr(admin_api, "REPOS", {"test_repo": test_repo_path})
    monkeypatch.setattr(server_state, "_INDEX", idx)
    monkeypatch.setattr(server_state, "_WATCHER", None)

    client = TestClient(admin_api.app)
    try:
        yield {"client": client, "index": idx, "config": cfg, "log_path": log_path}
    finally:
        idx.close()


def test_admin_status_reports_index(admin_app):
    client = admin_app["client"]
    resp = client.get("/admin/status")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["index"]["path"]
    assert payload["index"]["trigram_backend"] == "rocksdict"
    assert payload["repos"]["test_repo"] == str(server_state.REPOS["test_repo"])


def test_admin_index_rebuild_and_stats(admin_app):
    client = admin_app["client"]
    rebuild = client.post("/admin/index/rebuild", json={"repo": "test_repo", "force": True})
    assert rebuild.status_code == 200
    assert rebuild.json().get("files_indexed", 0) >= 1

    stats = client.get("/admin/index/stats", params={"repo": "test_repo"})
    assert stats.status_code == 200
    body = stats.json()
    assert body["repos"]["test_repo"]["documents"] >= 1


def test_admin_logs_tail_reads_file(admin_app):
    client = admin_app["client"]
    resp = client.get("/admin/logs/tail", params={"n": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"].endswith("server.log")
    assert body["lines"][-1].strip() == "line3"


def test_admin_index_file_rebuild_unknown_repo(admin_app):
    client = admin_app["client"]
    resp = client.post("/admin/index/file/rebuild", json={"repo": "missing", "path": "foo.py"})
    assert resp.status_code == 404
    assert resp.json()["error"] == "unknown_repo"


def test_admin_config_update_forbidden_in_prod(admin_app):
    client = admin_app["client"]
    cfg = admin_app["config"]
    cfg._mode = "prod"
    cfg.config_data["mode"] = "prod"
    cfg.config_data.setdefault("admin", {})["require_api_key"] = True
    cfg.config_data["admin"]["api_key"] = "secret"

    resp = client.post("/admin/config", content=b"{}", headers={"x-admin-key": "secret"})
    assert resp.status_code == 403  # prod mode disallows config edits


def test_admin_config_update_invalid_json(admin_app):
    client = admin_app["client"]
    resp = client.post(
        "/admin/config",
        content=b"{not-json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_json"


def test_admin_set_repo_gitignore_updates_options(admin_app, monkeypatch):
    client = admin_app["client"]
    saved = {}

    def fake_save(cfg):
        saved["data"] = cfg.config_data
        return Path("/tmp/config.json")

    monkeypatch.setattr(admin_api, "save_config", fake_save)

    resp = client.post(
        "/admin/repo/test_repo/gitignore",
        json={"respect_gitignore": False, "ignore_patterns": ["*.tmp"]},
    )
    assert resp.status_code == 200
    assert resp.json()["respect_gitignore"] is False
    assert "test_repo" in server_state.REPO_OPTIONS
    assert server_state.REPO_OPTIONS["test_repo"]["respect_gitignore"] is False
    assert "*.tmp" in server_state.REPO_OPTIONS["test_repo"].get("ignore_patterns", [])
    assert saved.get("data")


def test_admin_set_repo_include_solution(admin_app, monkeypatch):
    client = admin_app["client"]
    resp = client.post(
        "/admin/repo/test_repo/embeddings_include_solution",
        json={"embeddings_include_solution": True},
    )
    assert resp.status_code == 200
    assert server_state.REPO_OPTIONS["test_repo"]["embeddings_include_solution"] is True


def test_admin_index_stale_and_hardwrap_report(admin_app):
    client = admin_app["client"]
    stale = client.get("/admin/index/stale", params={"repo": "test_repo"})
    assert stale.status_code == 200
    body = stale.json()
    assert body["success"] is True
    assert "test_repo" in body["repos"]

    report = client.get("/admin/report/hardwraps", params={"repo": "test_repo"})
    assert report.status_code == 200
    assert report.json()["success"] is True


def test_admin_forbidden_ip(admin_app):
    client = admin_app["client"]
    cfg = admin_app["config"]
    cfg.config_data["admin"]["allowed_ips"] = []
    resp = client.get("/admin/status")
    assert resp.status_code == 403
    assert resp.json()["error"] == "forbidden"


def test_admin_requires_api_key_in_prod(admin_app):
    client = admin_app["client"]
    cfg = admin_app["config"]
    cfg._mode = "prod"
    cfg.config_data["mode"] = "prod"
    cfg.config_data["admin"].pop("api_key", None)
    resp = client.get("/admin/status")
    assert resp.status_code == 503
    assert resp.json()["error"] == "admin_api_key_required"


def test_admin_invalid_api_key_rejected(admin_app):
    client = admin_app["client"]
    cfg = admin_app["config"]
    cfg.config_data["admin"]["require_api_key"] = True
    cfg.config_data["admin"]["api_key"] = "secret"
    resp = client.get("/admin/status", headers={"X-Admin-Key": "wrong"})
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


def test_admin_optional_api_key_mismatch(admin_app):
    client = admin_app["client"]
    cfg = admin_app["config"]
    cfg.config_data["admin"]["require_api_key"] = False
    cfg.config_data["admin"]["api_key"] = "secret"
    resp = client.get("/admin/status", headers={"X-Admin-Key": "wrong"})
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


def test_admin_logs_tail_missing_file(admin_app):
    client = admin_app["client"]
    log_path = admin_app["log_path"]
    log_path.unlink()
    resp = client.get("/admin/logs/tail", params={"n": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert "Log file not found" in "\n".join(body["lines"])
    assert body["path"].endswith("server.log")


def test_admin_logs_tail_no_configured_path(admin_app, monkeypatch):
    client = admin_app["client"]
    # Force the helper to report no configured log file
    monkeypatch.setattr(
        "sigil_mcp.logging_setup.get_log_file_path",
        lambda _config_log=None: None,
    )
    resp = client.get("/admin/logs/tail")
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "N/A"
    assert "No log file configured" in "\n".join(body["lines"])


def test_admin_mcp_refresh_missing_manager(admin_app, monkeypatch):
    client = admin_app["client"]
    monkeypatch.setattr(server_state, "get_global_manager", lambda: None)
    resp = client.post("/admin/mcp/refresh")
    assert resp.status_code == 500
    payload = resp.json()
    assert payload["error"] == "refresh_failed"
    assert "not initialized" in payload["detail"]


def test_admin_root_endpoint(admin_app):
    resp = admin_app["client"].get("/admin")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "Sigil MCP Admin API"
    assert "endpoints" in body


def test_admin_index_file_rebuild_missing_fields(admin_app):
    client = admin_app["client"]
    resp = client.post("/admin/index/file/rebuild", json={"repo": "test_repo"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_request"


def test_admin_get_repo_unknown_repo(admin_app):
    client = admin_app["client"]
    resp = client.get("/admin/repo/unknown")
    assert resp.status_code == 404
    assert resp.json()["error"] == "repo_not_found"


def test_admin_get_repo_reads_repo_without_options(admin_app):
    client = admin_app["client"]
    server_state.REPO_OPTIONS.pop("test_repo", None)
    resp = client.get("/admin/repo/test_repo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "test_repo"
    assert body["path"] == str(server_state.REPOS["test_repo"])


def test_admin_status_when_no_repos(monkeypatch, admin_app):
    client = admin_app["client"]
    monkeypatch.setattr(admin_api, "REPOS", {})
    monkeypatch.setattr(server_state, "REPOS", {})
    resp = client.get("/admin/status")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["repos"] == {}
    assert payload["index"]["path"] is None


def test_admin_config_update_invalid_payload(admin_app):
    client = admin_app["client"]
    resp = client.post("/admin/config", json=["not-a-dict"])
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_payload"


def test_admin_config_update_save_failure(monkeypatch, admin_app):
    client = admin_app["client"]
    monkeypatch.setattr(admin_api, "save_config", lambda cfg: (_ for _ in ()).throw(RuntimeError("fail")))
    resp = client.post("/admin/config", json={"admin": {"enabled": True}})
    assert resp.status_code == 500
    assert resp.json()["error"] == "config_save_failed"


def test_admin_index_stats_database_locked(monkeypatch, admin_app):
    client = admin_app["client"]
    monkeypatch.setattr(
        admin_api,
        "get_index_stats_op",
        lambda repo=None: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )
    resp = client.get("/admin/index/stats")
    assert resp.status_code == 503
    assert resp.json()["error"] == "database_locked"


def test_admin_index_rebuild_internal_error(monkeypatch, admin_app):
    client = admin_app["client"]
    monkeypatch.setattr(admin_api, "rebuild_index_op", lambda repo, force: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.post("/admin/index/rebuild", json={"repo": None, "force": True})
    assert resp.status_code == 500
    assert resp.json()["error"] == "internal_error"


def test_admin_vector_rebuild_handles_locked(monkeypatch, admin_app):
    client = admin_app["client"]

    async def no_sleep(*_args, **_kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        admin_api,
        "build_vector_index_op",
        lambda repo, force, model=None: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )
    resp = client.post("/admin/vector/rebuild", json={"repo": None, "force": True})
    assert resp.status_code == 503
    assert resp.json()["error"] == "database_locked"


def test_admin_logs_tail_handles_read_error(monkeypatch, admin_app):
    client = admin_app["client"]

    def fail_open(*_args, **_kwargs):
        raise OSError("cannot read")

    monkeypatch.setattr(Path, "open", fail_open)
    resp = client.get("/admin/logs/tail")
    assert resp.status_code == 500
    assert resp.json()["error"] == "internal_error"


def test_admin_index_file_rebuild_success(admin_app):
    client = admin_app["client"]
    resp = client.post("/admin/index/file/rebuild", json={"repo": "test_repo", "path": "main.py"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["indexed"] in {True, False}


def test_admin_vector_rebuild_success(admin_app):
    client = admin_app["client"]
    resp = client.post("/admin/vector/rebuild", json={"repo": "test_repo", "force": False, "model": "default"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"completed", "skipped"}


def test_admin_config_update_success(monkeypatch, admin_app):
    client = admin_app["client"]
    saved = {}

    def fake_save(cfg):
        saved["path"] = Path("/tmp/config.json")
        return saved["path"]

    def fake_load(path):
        saved["loaded"] = True
        return admin_app["config"]

    monkeypatch.setattr(admin_api, "save_config", fake_save)
    monkeypatch.setattr(admin_api, "load_config", fake_load)
    resp = client.post("/admin/config", json={"admin": {"enabled": True}})
    assert resp.status_code == 200
    assert saved["path"]
    assert saved["loaded"]


def test_admin_index_stats_generic_error(monkeypatch, admin_app):
    client = admin_app["client"]
    monkeypatch.setattr(admin_api, "get_index_stats_op", lambda repo=None: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.get("/admin/index/stats")
    assert resp.status_code == 500
    assert resp.json()["error"] == "internal_error"
