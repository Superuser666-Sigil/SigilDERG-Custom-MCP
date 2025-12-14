from starlette.requests import Request

import sigil_mcp.admin_api as admin_api


def _make_request(client_ip="1.2.3.4", headers=None):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers or [],
        "client": (client_ip, 1234),
        "server": ("test", 80),
        "scheme": "http",
    }
    return Request(scope)


def test_require_admin_prod_without_api_key(monkeypatch):
    monkeypatch.setattr(
        admin_api,
        "_get_admin_cfg",
        lambda: {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 9000,
            "api_key": None,
            "require_api_key": True,
            "allowed_ips": ["1.2.3.4"],
            "mode": "prod",
        },
    )
    resp = admin_api.require_admin(_make_request())
    assert resp is not None
    result = admin_api.asyncio.run(resp)
    assert result.status_code == 503
    assert "admin_api_key_required" in result.body.decode()


def test_require_admin_optional_api_key_mismatch(monkeypatch):
    monkeypatch.setattr(
        admin_api,
        "_get_admin_cfg",
        lambda: {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 9000,
            "api_key": "secret",
            "require_api_key": False,
            "allowed_ips": ["1.2.3.4"],
            "mode": "dev",
        },
    )
    req = _make_request(headers=[(b"x-admin-key", b"wrong")])
    resp = admin_api.require_admin(req)
    assert resp is not None
    result = admin_api.asyncio.run(resp)
    assert result.status_code == 401


def test_is_allowed_ip_defaults(monkeypatch):
    monkeypatch.setattr(
        admin_api,
        "_get_admin_cfg",
        lambda: {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 9000,
            "api_key": None,
            "require_api_key": True,
            "allowed_ips": [],
            "mode": "dev",
        },
    )
    assert admin_api._is_allowed_ip(None) is False
    # Falls back to localhost allow-list when allowed_ips empty
    assert admin_api._is_allowed_ip("127.0.0.1") is True
