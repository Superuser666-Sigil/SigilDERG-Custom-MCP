# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

from sigil_mcp.security import auth as security_auth


def test_api_key_is_valid_prefers_env(monkeypatch):
    monkeypatch.setenv("SIGIL_MCP_API_KEY", "env-key")
    called = {"verify": False}
    monkeypatch.setattr(security_auth, "verify_api_key", lambda key: called.__setitem__("verify", True))

    assert security_auth.api_key_is_valid("env-key") is True
    # verify_api_key should not be called when env matches
    assert called["verify"] is False


def test_check_authentication_whitelist_requires_ip(monkeypatch):
    settings = security_auth.AuthSettings(
        auth_enabled=False,
        oauth_enabled=False,
        allow_local_bypass=False,
        allowed_ips=("10.0.0.1",),
        mode="dev",
    )
    assert security_auth.check_authentication(settings=settings) is False


def test_check_authentication_accepts_valid_api_key(monkeypatch):
    settings = security_auth.AuthSettings(
        auth_enabled=True,
        oauth_enabled=False,
        allow_local_bypass=False,
        allowed_ips=(),
        mode="dev",
    )
    monkeypatch.setattr(security_auth, "api_key_is_valid", lambda key: True)
    assert security_auth.check_authentication(
        request_headers={"x-api-key": "abc"}, client_ip="127.0.0.1", settings=settings
    )


def test_is_redirect_uri_allowed_allow_list():
    assert security_auth.is_redirect_uri_allowed(
        "https://example.com/callback",
        registered_redirects=[],
        allow_list=["https://example.com/"],
    )
