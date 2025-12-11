# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import subprocess

import sigil_mcp.mcp_installer as installer


def test_auto_install_runs_known_command(monkeypatch):
    servers = [{"name": "s1", "command": "npm", "args": ["i"], "auto_install": True}]
    called = {"ran": False}

    def fake_run(cmd, check):
        called["ran"] = True
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "run", fake_run)
    installer.auto_install(servers)
    assert called["ran"] is True
