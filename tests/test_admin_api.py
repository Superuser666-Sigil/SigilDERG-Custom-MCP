import os
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient
from sigil_mcp.admin_api import app

def test_admin_api_disabled():
    with patch("sigil_mcp.admin_api._get_admin_cfg") as mock_cfg:
        mock_cfg.return_value = {"enabled": False}
        client = TestClient(app)
        response = client.get("/admin/status")
        assert response.status_code == 503
        assert response.json() == {"error": "admin_disabled"}

def test_admin_api_status():
    with patch("sigil_mcp.admin_api._get_admin_cfg") as mock_cfg:
        mock_cfg.return_value = {
            "enabled": True, 
            "host": "127.0.0.1", 
            "port": 8765, 
            "api_key": None,
            "allowed_ips": ["127.0.0.1", "testclient"]
        }
        client = TestClient(app)
        response = client.get("/admin/status")
        assert response.status_code == 200
        data = response.json()
        assert data["admin"]["enabled"] is True
        assert "repos" in data

def test_admin_api_auth_required():
    with patch("sigil_mcp.admin_api._get_admin_cfg") as mock_cfg:
        mock_cfg.return_value = {
            "enabled": True, 
            "host": "127.0.0.1", 
            "port": 8765, 
            "api_key": "secret",
            "allowed_ips": ["127.0.0.1", "testclient"]
        }
        client = TestClient(app)
        
        # No key
        response = client.get("/admin/status")
        assert response.status_code == 401
        
        # Wrong key
        response = client.get("/admin/status", headers={"X-Admin-Key": "wrong"})
        assert response.status_code == 401
        
        # Correct key
        response = client.get("/admin/status", headers={"X-Admin-Key": "secret"})
        assert response.status_code == 200

def test_admin_config_view():
    with patch("sigil_mcp.admin_api._get_admin_cfg") as mock_cfg:
        mock_cfg.return_value = {
            "enabled": True, 
            "host": "127.0.0.1", 
            "port": 8765, 
            "api_key": None,
            "allowed_ips": ["127.0.0.1", "testclient"]
        }
        client = TestClient(app)
        response = client.get("/admin/config")
        assert response.status_code == 200
        # config.json structure check - verify it's a dict with expected keys
        config_data = response.json()
        assert isinstance(config_data, dict)
        # Check for at least one expected config section
        assert any(key in config_data for key in ["server", "repositories", "watch", "index"])
