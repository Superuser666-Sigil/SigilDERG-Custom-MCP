# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import asyncio
import sqlite3
from pathlib import Path

import httpx
import pytest

import sigil_mcp.admin_api as admin_api


class _DummyLoop:
    def __init__(self, exc: Exception | None = None):
        self._exc = exc

    async def run_in_executor(self, _executor, func, *args):
        if self._exc:
            raise self._exc
        return func(*args)


def _cfg():
    return {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 8765,
        "api_key": None,
        "require_api_key": False,
        "allowed_ips": ["127.0.0.1"],
        "mode": "dev",
    }


@pytest.mark.anyio
async def test_admin_index_rebuild_success(monkeypatch):
    monkeypatch.setattr(admin_api, "_get_admin_cfg", lambda: _cfg())
    monkeypatch.setattr(
        admin_api, "rebuild_index_op", lambda repo, force: {"repo": repo, "force": force}
    )
    monkeypatch.setattr(admin_api.asyncio, "get_event_loop", lambda: _DummyLoop())

    transport = httpx.ASGITransport(app=admin_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/admin/index/rebuild", json={"repo": "r1", "force": False})

    assert resp.status_code == 200
    assert resp.json() == {"repo": "r1", "force": False}


@pytest.mark.anyio
async def test_admin_vector_rebuild_handles_locked_db(monkeypatch):
    monkeypatch.setattr(admin_api, "_get_admin_cfg", lambda: _cfg())
    monkeypatch.setattr(admin_api, "build_vector_index_op", lambda repo, force, model: {})
    monkeypatch.setattr(
        admin_api.asyncio,
        "get_event_loop",
        lambda: _DummyLoop(sqlite3.OperationalError("database is locked")),
    )

    transport = httpx.ASGITransport(app=admin_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/admin/vector/rebuild", json={"repo": "r1", "force": True})

    assert resp.status_code == 503
    payload = resp.json()
    assert payload["error"] == "database_locked"


@pytest.mark.anyio
async def test_admin_index_stats_calls_op(monkeypatch):
    monkeypatch.setattr(admin_api, "_get_admin_cfg", lambda: _cfg())
    monkeypatch.setattr(admin_api, "get_index_stats_op", lambda repo: {"repo": repo or "all"})
    monkeypatch.setattr(admin_api.asyncio, "get_event_loop", lambda: _DummyLoop())

    transport = httpx.ASGITransport(app=admin_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/admin/index/stats")

    assert resp.status_code == 200
    assert resp.json() == {"repo": "all"}


@pytest.mark.anyio
async def test_admin_mcp_status(monkeypatch):
    monkeypatch.setattr(admin_api, "_get_admin_cfg", lambda: _cfg())
    monkeypatch.setattr(admin_api, "external_mcp_status_op", lambda: {"enabled": True})

    transport = httpx.ASGITransport(app=admin_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/admin/mcp/status")

    assert resp.status_code == 200
    assert resp.json()["enabled"] is True


@pytest.mark.anyio
async def test_admin_logs_tail_no_file(monkeypatch, tmp_path: Path):
    log_path = tmp_path / "missing.log"
    monkeypatch.setenv("SIGIL_MCP_LOG_FILE", str(log_path))
    monkeypatch.setattr(admin_api, "_get_admin_cfg", lambda: _cfg())

    transport = httpx.ASGITransport(app=admin_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/admin/logs/tail")

    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == str(log_path)
    assert "Log file not found" in "\n".join(body["lines"])
