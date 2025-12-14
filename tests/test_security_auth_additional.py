# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).

import secrets
import sys
import types

from sigil_mcp.security import auth as sec_auth


def test_redirect_uri_not_allowed():
    assert sec_auth.is_redirect_uri_allowed(
        "https://example.com/callback",
        registered_redirects=[],
        allow_list=[],
    ) is False


def test_check_authentication_ip_whitelist_and_invalid_token(monkeypatch):
    settings = sec_auth.AuthSettings(
        auth_enabled=True,
        oauth_enabled=True,
        allow_local_bypass=False,
        allowed_ips=("10.0.0.1",),
        mode="dev",
    )
    # missing IP when whitelist enforced -> False
    assert sec_auth.check_authentication(settings=settings, request_headers={}) is False
    # IP not allowed -> False
    assert (
        sec_auth.check_authentication(
            settings=settings, request_headers={}, client_ip="8.8.8.8"
        )
        is False
    )

    # Allowed IP but invalid bearer token path
    import sys
    dummy_mgr = types.SimpleNamespace(verify_token=lambda token: False)
    dummy_module = types.SimpleNamespace(get_oauth_manager=lambda: dummy_mgr)
    monkeypatch.setitem(sys.modules, "sigil_mcp.oauth", dummy_module)
    headers = {"Authorization": "Bearer invalid"}
    assert (
        sec_auth.check_authentication(
            settings=settings, request_headers=headers, client_ip="10.0.0.1"
        )
        is False
    )


def test_check_ip_whitelist_empty_and_mismatch():
    settings = sec_auth.AuthSettings(
        auth_enabled=True,
        oauth_enabled=False,
        allow_local_bypass=False,
        allowed_ips=(),
        mode="dev",
    )
    assert sec_auth.check_ip_whitelist(settings=settings) is True
    settings2 = sec_auth.AuthSettings(
        auth_enabled=True,
        oauth_enabled=False,
        allow_local_bypass=False,
        allowed_ips=("1.2.3.4",),
        mode="dev",
    )
    assert sec_auth.check_ip_whitelist(settings=settings2, client_ip="2.2.2.2") is False


def test_get_auth_settings_default(monkeypatch):
    class DummyConfig:
        auth_enabled = True
        oauth_enabled = False
        allow_local_bypass = False
        allowed_ips = ["1.1.1.1"]
        mode = "prod"

    import sigil_mcp.config as cfg

    monkeypatch.setattr(cfg, "get_config", lambda: DummyConfig())
    monkeypatch.setitem(sys.modules, "sigil_mcp.config", cfg)
    settings = sec_auth.get_auth_settings()
    assert settings.allowed_ips == ("1.1.1.1",)


def test_is_local_connection_variants():
    assert sec_auth.is_local_connection(None) is False
    assert sec_auth.is_local_connection("127.0.0.1") is True


def test_is_redirect_uri_allowed_localhost_http():
    assert sec_auth.is_redirect_uri_allowed("http://localhost/cb", [], []) is True


def test_api_key_is_valid_handles_compare_error(monkeypatch):
    monkeypatch.setattr(sec_auth, "get_api_key_from_env", lambda: "secret")
    monkeypatch.setattr(sec_auth, "verify_api_key", lambda k: False)

    def bad_compare(a, b):
        raise RuntimeError("boom")

    monkeypatch.setattr(secrets, "compare_digest", bad_compare)
    assert sec_auth.api_key_is_valid("secret") is False


def test_check_authentication_local_bypass_prod():
    settings = sec_auth.AuthSettings(
        auth_enabled=True,
        oauth_enabled=False,
        allow_local_bypass=True,
        allowed_ips=(),
        mode="prod",
    )
    assert sec_auth.check_authentication(settings=settings, client_ip="127.0.0.1") is True


def test_check_authentication_disabled_in_prod():
    settings = sec_auth.AuthSettings(
        auth_enabled=False,
        oauth_enabled=False,
        allow_local_bypass=False,
        allowed_ips=(),
        mode="prod",
    )
    assert sec_auth.check_authentication(settings=settings) is True


def test_check_authentication_oauth_valid(monkeypatch):
    class DummyOAuth:
        def verify_token(self, token):
            return True

    settings = sec_auth.AuthSettings(
        auth_enabled=True,
        oauth_enabled=True,
        allow_local_bypass=False,
        allowed_ips=(),
        mode="dev",
    )
    monkeypatch.setattr("sigil_mcp.oauth.get_oauth_manager", lambda: DummyOAuth())
    headers = {"Authorization": "Bearer tok"}
    assert sec_auth.check_authentication(request_headers=headers, settings=settings) is True


def test_check_ip_whitelist_denied():
    settings = sec_auth.AuthSettings(
        auth_enabled=True,
        oauth_enabled=False,
        allow_local_bypass=False,
        allowed_ips=("1.1.1.1",),
        mode="dev",
    )
    assert sec_auth.check_ip_whitelist(client_ip="2.2.2.2", settings=settings) is False


def test_check_authentication_local_bypass_dev():
    settings = sec_auth.AuthSettings(
        auth_enabled=True,
        oauth_enabled=False,
        allow_local_bypass=True,
        allowed_ips=(),
        mode="dev",
    )
    assert sec_auth.check_authentication(settings=settings, client_ip="127.0.0.1") is True


def test_check_authentication_disabled_in_dev():
    settings = sec_auth.AuthSettings(
        auth_enabled=False,
        oauth_enabled=False,
        allow_local_bypass=False,
        allowed_ips=(),
        mode="dev",
    )
    assert sec_auth.check_authentication(settings=settings) is True


def test_check_authentication_oauth_without_header():
    settings = sec_auth.AuthSettings(
        auth_enabled=True,
        oauth_enabled=True,
        allow_local_bypass=False,
        allowed_ips=(),
        mode="dev",
    )
    assert sec_auth.check_authentication(request_headers={"foo": "bar"}, settings=settings) is False


def test_check_authentication_invalid_api_key_logs(monkeypatch):
    settings = sec_auth.AuthSettings(
        auth_enabled=True,
        oauth_enabled=False,
        allow_local_bypass=False,
        allowed_ips=(),
        mode="dev",
    )
    monkeypatch.setattr(sec_auth, "api_key_is_valid", lambda key: False)
    assert sec_auth.check_authentication(request_headers={"X-API-Key": "bad"}, settings=settings) is False


def test_check_ip_whitelist_accepts_match():
    settings = sec_auth.AuthSettings(
        auth_enabled=True,
        oauth_enabled=False,
        allow_local_bypass=False,
        allowed_ips=("10.0.0.1",),
        mode="dev",
    )
    assert sec_auth.check_ip_whitelist(client_ip="10.0.0.1", settings=settings) is True
