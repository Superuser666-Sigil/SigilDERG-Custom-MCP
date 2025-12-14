# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import asyncio
from unittest.mock import patch

import httpx

from sigil_mcp.admin_api import app


def _base_cfg():
    return {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 8765,
        "api_key": None,
        "require_api_key": False,
        "allowed_ips": ["127.0.0.1", "testclient"],
    }


async def _request(path: str, headers: dict | None = None):
    """Issue an HTTP request against the Admin API using an ASGI transport."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        return await client.get(path, headers=headers)


def test_admin_api_disabled():
    with patch("sigil_mcp.admin_api._get_admin_cfg") as mock_cfg:
        mock_cfg.return_value = {"enabled": False}
        response = asyncio.run(_request("/admin/status"))
        assert response.status_code == 503
        assert response.json() == {"error": "admin_disabled"}


def test_admin_api_status():
    with patch("sigil_mcp.admin_api._get_admin_cfg") as mock_cfg:
        cfg = _base_cfg()
        mock_cfg.return_value = cfg
        response = asyncio.run(_request("/admin/status"))
        assert response.status_code == 200
        data = response.json()
        assert data["admin"]["enabled"] is True
        assert "repos" in data


def test_admin_api_auth_required():
    with patch("sigil_mcp.admin_api._get_admin_cfg") as mock_cfg:
        cfg = _base_cfg()
        cfg["api_key"] = "secret"
        cfg["require_api_key"] = True
        mock_cfg.return_value = cfg

        # No key
        response = asyncio.run(_request("/admin/status"))
        assert response.status_code == 401

        # Wrong key
        response = asyncio.run(_request("/admin/status", headers={"X-Admin-Key": "wrong"}))
        assert response.status_code == 401

        # Correct key
        response = asyncio.run(_request("/admin/status", headers={"X-Admin-Key": "secret"}))
        assert response.status_code == 200


def test_admin_api_missing_required_key_returns_503():
    with patch("sigil_mcp.admin_api._get_admin_cfg") as mock_cfg:
        cfg = _base_cfg()
        cfg["api_key"] = None
        cfg["require_api_key"] = True
        mock_cfg.return_value = cfg
        response = asyncio.run(_request("/admin/status"))
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "configuration_error"


def test_admin_config_view():
    with patch("sigil_mcp.admin_api._get_admin_cfg") as mock_cfg:
        cfg = _base_cfg()
        mock_cfg.return_value = cfg
        response = asyncio.run(_request("/admin/config"))
        assert response.status_code == 200
        # config.json structure check - verify it's a dict with expected keys
        config_data = response.json()
        assert isinstance(config_data, dict)
        # Check for at least one expected config section
        assert any(key in config_data for key in ["server", "repositories", "watch", "index"])
