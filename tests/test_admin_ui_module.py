# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import subprocess
from pathlib import Path

import sigil_mcp.admin_ui as admin_ui


class DummyCfg:
    def __init__(self, enabled=True, autostart=True, path=None):
        self.admin_enabled = enabled
        self.admin_ui_auto_start = autostart
        self.admin_ui_path = path or "."
        self.admin_ui_command = "npm"
        self.admin_ui_args = ["run", "dev"]
        self.admin_ui_port = 5173
        self.server_host = "127.0.0.1"
        self.server_port = 8000


def test_admin_ui_skip_when_disabled(monkeypatch):
    monkeypatch.setattr(admin_ui, "get_config", lambda: DummyCfg(enabled=False))
    called = {"popen": False}
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: called.__setitem__("popen", True))
    admin_ui.start_admin_ui()
    assert called["popen"] is False


def test_admin_ui_start_missing_command(monkeypatch, tmp_path):
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    cfg = DummyCfg(path=str(ui_dir))
    monkeypatch.setattr(admin_ui, "get_config", lambda: cfg)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    admin_ui.start_admin_ui()
    # No exception and process remains None
    assert admin_ui._ADMIN_UI_PROCESS.proc is None


def test_admin_ui_start_and_stop(monkeypatch, tmp_path):
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    cfg = DummyCfg(path=str(ui_dir))
    monkeypatch.setattr(admin_ui, "get_config", lambda: cfg)
    fake_proc = type("P", (), {"pid": 1, "terminate": lambda self: None, "wait": lambda self, timeout=5: None})
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: fake_proc())
    admin_ui.start_admin_ui()
    assert admin_ui._ADMIN_UI_PROCESS.proc is not None
    admin_ui.stop_admin_ui()
    assert admin_ui._ADMIN_UI_PROCESS.proc is None
