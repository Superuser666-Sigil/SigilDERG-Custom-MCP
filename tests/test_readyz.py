# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import importlib
from pathlib import Path


def test_readyz_reports_ready_when_embeddings_disabled(monkeypatch):
    index_root = Path("tmp_index_readyz").resolve()
    index_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(index_root)
    monkeypatch.setenv("SIGIL_INDEX_PATH", str(index_root))
    monkeypatch.setenv("SIGIL_MCP_MODE", "dev")
    monkeypatch.setenv("SIGIL_MCP_WATCH_ENABLED", "false")
    log_file = index_root / "server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SIGIL_MCP_LOG_FILE", str(log_file))

    server = importlib.reload(importlib.import_module("sigil_mcp.server"))

    readiness = server._is_ready()

    assert readiness["config"] is True
    assert readiness["index"] is True
    assert readiness["embeddings"] is True

    # Clean up connections to avoid locking temp dirs during test runs
    if getattr(server, "_INDEX", None):
        if getattr(server._INDEX, "close", None):
            server._INDEX.close()
    # Optionally, reload server to clear state for other tests
    importlib.reload(server)
