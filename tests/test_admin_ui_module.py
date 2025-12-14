# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import subprocess

import sigil_mcp.admin_ui as admin_ui


class DummyCfg:
    def __init__(self, enabled=True, autostart=True, path=None):
        self.admin_enabled = enabled
        self.admin_ui_auto_start = autostart
        self.admin_ui_path = path or "."
        self.admin_ui_command = "npm"
        self.admin_ui_args = ["run", "dev"]
        self.admin_ui_port = 5173
        self.admin_host = "127.0.0.1"
        self.admin_port = 9000
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


def test_admin_ui_start_generic_exception(monkeypatch, tmp_path):
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    cfg = DummyCfg(path=str(ui_dir))
    monkeypatch.setattr(admin_ui, "_ADMIN_UI_PROCESS", admin_ui.AdminUIProcess())
    monkeypatch.setattr(admin_ui, "get_config", lambda: cfg)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    admin_ui.start_admin_ui()
    assert admin_ui._ADMIN_UI_PROCESS.proc is None


def test_admin_ui_stop_handles_errors(monkeypatch):
    class BadProc:
        def terminate(self):
            raise RuntimeError("fail")

        def wait(self, timeout=5):
            raise RuntimeError("fail")

        def kill(self):
            raise RuntimeError("fail")

    proc = BadProc()
    monkeypatch.setattr(admin_ui, "_ADMIN_UI_PROCESS", admin_ui.AdminUIProcess())
    admin_ui._ADMIN_UI_PROCESS.proc = proc
    admin_ui.stop_admin_ui()
    assert admin_ui._ADMIN_UI_PROCESS.proc is None


def test_start_stop_admin_ui_wrappers_handle_exceptions(monkeypatch):
    class FailingProcess:
        def start(self):
            raise RuntimeError("boom")

        def stop(self):
            raise RuntimeError("stop-fail")

    monkeypatch.setattr(admin_ui, "_ADMIN_UI_PROCESS", FailingProcess())
    admin_ui.start_admin_ui()
    admin_ui.stop_admin_ui()


def test_admin_ui_stop_no_proc(monkeypatch):
    monkeypatch.setattr(admin_ui, "_ADMIN_UI_PROCESS", admin_ui.AdminUIProcess())
    admin_ui.stop_admin_ui()


def test_admin_ui_autostart_disabled(monkeypatch, tmp_path):
    cfg = DummyCfg(autostart=False, path=str(tmp_path))
    monkeypatch.setattr(admin_ui, "get_config", lambda: cfg)
    called = {"popen": False}
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: called.__setitem__("popen", True))
    admin_ui.start_admin_ui()
    assert called["popen"] is False


def test_admin_ui_path_missing(monkeypatch, tmp_path):
    cfg = DummyCfg(path=str(tmp_path / "missing"))
    monkeypatch.setattr(admin_ui, "get_config", lambda: cfg)
    called = {"popen": False}
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: called.__setitem__("popen", True))
    admin_ui.start_admin_ui()
    assert called["popen"] is False


def test_admin_ui_env_points_to_admin_backend(monkeypatch, tmp_path):
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    cfg = DummyCfg(path=str(ui_dir))
    cfg.admin_host = "admin.local"
    cfg.admin_port = 1234
    captured = {}

    def fake_popen(cmd, cwd=None, env=None, stdout=None, stderr=None):
        captured["env"] = env
        class P:
            def __init__(self):
                self.pid = 1
            def terminate(self): pass
            def wait(self, timeout=5): pass
        return P()

    monkeypatch.setattr(admin_ui, "get_config", lambda: cfg)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    admin_ui.start_admin_ui()
    env = captured["env"]
    assert env["VITE_API_BASE_URL"] == "http://admin.local:1234"
    assert env["PORT"] == str(cfg.admin_ui_port)
    admin_ui.stop_admin_ui()
