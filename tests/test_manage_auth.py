# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import os
from pathlib import Path

import pytest

from sigil_mcp import manage_auth
from sigil_mcp.auth import get_api_key_path


@pytest.fixture(autouse=True)
def temp_api_key_path(monkeypatch, tmp_path):
    path = tmp_path / "api_key"
    monkeypatch.setenv("SIGIL_MCP_API_KEY_FILE", str(path))
    # reset cached path
    from sigil_mcp import auth as auth_mod
    auth_mod._api_key_path = None
    auth_mod.API_KEY_FILE = auth_mod.get_api_key_path()
    yield path
    if path.exists():
        path.unlink()


def test_generate_creates_key(capsys):
    rc = manage_auth.generate()
    assert rc == 0
    key_path = get_api_key_path()
    assert key_path.exists()
    out = capsys.readouterr().out
    assert "API Key Generated Successfully" in out


def test_generate_noop_when_exists(capsys, temp_api_key_path):
    temp_api_key_path.write_text("dummy")
    rc = manage_auth.generate()
    assert rc == 1
    out = capsys.readouterr().out
    assert "already exists" in out


def test_show_without_key(capsys, temp_api_key_path):
    if temp_api_key_path.exists():
        temp_api_key_path.unlink()
    rc = manage_auth.show()
    assert rc == 1
    assert "No API key configured" in capsys.readouterr().out


def test_reset_generates_new_key(monkeypatch, capsys, temp_api_key_path):
    temp_api_key_path.write_text("oldhash")
    monkeypatch.setenv("SIGIL_MCP_API_KEY_FILE", str(temp_api_key_path))
    monkeypatch.setattr("builtins.input", lambda _prompt="": "yes")
    from sigil_mcp import auth as auth_mod
    auth_mod._api_key_path = None
    auth_mod.API_KEY_FILE = auth_mod.get_api_key_path()

    rc = manage_auth.reset()
    assert rc == 0
    new_out = capsys.readouterr().out
    assert "Deleted old API key" in new_out
    assert temp_api_key_path.exists()
