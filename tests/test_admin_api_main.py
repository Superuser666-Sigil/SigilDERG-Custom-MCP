# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import importlib
import sys

import pytest


def test_admin_api_main_exits_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGIL_MCP_LOG_FILE", str(tmp_path / "log.log"))
    mock_cfg = type(
        "Cfg",
        (),
        {
            "admin_enabled": False,
            "log_level": "INFO",
            "log_file": str(tmp_path / "log.log"),
            "admin_host": "127.0.0.1",
            "admin_port": 8765,
            "admin_api_key": None,
            "admin_allowed_ips": [],
        },
    )
    import sigil_mcp.admin_api_main as admin_api_main

    monkeypatch.setattr(admin_api_main, "get_config", lambda: mock_cfg)
    monkeypatch.setattr(admin_api_main, "setup_logging", lambda *a, **k: None)

    with pytest.raises(SystemExit):
        admin_api_main.main()


def test_admin_api_main_runs_uvicorn(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGIL_MCP_LOG_FILE", str(tmp_path / "log.log"))
    mock_cfg = type(
        "Cfg",
        (),
        {
            "admin_enabled": True,
            "log_level": "INFO",
            "log_file": str(tmp_path / "log.log"),
            "admin_host": "127.0.0.1",
            "admin_port": 8765,
            "admin_api_key": "k",
            "admin_allowed_ips": ["127.0.0.1"],
        },
    )
    import sigil_mcp.admin_api_main as admin_api_main

    # Reload to ensure env takes effect
    admin_api_main = importlib.reload(admin_api_main)
    monkeypatch.setattr(admin_api_main, "get_config", lambda: mock_cfg)
    monkeypatch.setattr(admin_api_main, "setup_logging", lambda *a, **k: None)
    called = {}

    def _fake_run(app, host, port, log_level, access_log):
        called.update({"host": host, "port": port, "log_level": log_level, "access_log": access_log})

    monkeypatch.setattr(admin_api_main.uvicorn, "run", _fake_run)
    admin_api_main.main()
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8765
