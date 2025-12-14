# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import subprocess

import sigil_mcp.mcp_installer as installer


def test_auto_install_skips_disabled_and_unknown(monkeypatch):
    calls = []

    def fake_run(cmdline, check):
        calls.append(cmdline)

    monkeypatch.setattr(subprocess, "run", fake_run)
    servers = [
        {"name": "disabled", "disabled": True, "command": "npm"},
        {"name": "unknown", "command": "echo", "args": ["hi"]},
    ]
    installer.auto_install(servers)
    assert calls == []


def test_auto_install_handles_missing_command(monkeypatch):
    calls = []

    def fake_run(cmdline, check):
        calls.append(cmdline)
        raise FileNotFoundError("missing")

    monkeypatch.setattr(subprocess, "run", fake_run)
    installer.auto_install([{"name": "test", "command": "npm"}])
    assert calls[0][0] == "npm"


def test_auto_install_handles_called_process_error(monkeypatch):
    errors = []

    def fake_run(cmdline, check):
        errors.append("called")
        raise subprocess.CalledProcessError(returncode=1, cmd=cmdline)

    monkeypatch.setattr(subprocess, "run", fake_run)
    installer.auto_install([{"name": "test", "command": "npm"}])
    assert errors == ["called"]


def test_auto_install_handles_generic_exception(monkeypatch):
    errors = []

    def fake_run(cmdline, check):
        errors.append("err")
        raise RuntimeError("boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    installer.auto_install([{"name": "test", "command": "npm"}])
    assert errors == ["err"]


def test_auto_install_skips_missing_command(monkeypatch):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(True))
    installer.auto_install([{"name": "no-cmd"}])
    assert called == []
