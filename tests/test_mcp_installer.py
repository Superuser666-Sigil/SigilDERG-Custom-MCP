# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import subprocess
import sigil_mcp.mcp_installer as installer


def test_auto_install_runs_known_commands(monkeypatch):
    calls = []

    def fake_run(cmdline, check):
        calls.append((tuple(cmdline), check))
        return subprocess.CompletedProcess(cmdline, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    servers = [
        {"name": "playwright", "command": "npx", "args": ["@playwright/mcp@latest"], "disabled": False},
        {"name": "noop", "command": "python", "args": ["tool.py"], "disabled": False},
        {"name": "explicit", "command": "python", "args": ["tool.py"], "auto_install": True},
    ]
    installer.auto_install(servers)

    assert calls[0][0] == ("npx", "@playwright/mcp@latest")
    assert calls[1][0] == ("python", "tool.py")
