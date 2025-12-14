# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import asyncio
import importlib
from pathlib import Path

from starlette.requests import Request


def test_admin_api_rejects_missing_key_in_prod(monkeypatch):
    index_root = Path("tmp_admin_api_index").resolve()
    index_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(index_root)
    monkeypatch.setenv("SIGIL_INDEX_PATH", str(index_root))
    monkeypatch.setenv("SIGIL_MCP_MODE", "prod")
    monkeypatch.setenv("SIGIL_MCP_WATCH_ENABLED", "false")
    log_file = index_root / "server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SIGIL_MCP_LOG_FILE", str(log_file))
    monkeypatch.delenv("SIGIL_MCP_ADMIN_API_KEY", raising=False)

    server = importlib.reload(importlib.import_module("sigil_mcp.server"))
    admin_module = importlib.reload(importlib.import_module("sigil_mcp.admin_api"))

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/admin/status",
        "raw_path": b"/admin/status",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "http",
        "query_string": b"",
        "root_path": "",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
    }
    request = Request(scope)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    response = loop.run_until_complete(admin_module.require_admin(request))

    assert response.status_code == 503
    if server._INDEX:
        server._INDEX.close()
