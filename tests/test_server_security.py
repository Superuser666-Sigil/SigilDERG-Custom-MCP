# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

from sigil_mcp.security import auth as security_auth


def _make_settings(**overrides):
    return security_auth.AuthSettings(
        auth_enabled=bool(overrides.pop("auth_enabled", True)),
        oauth_enabled=bool(overrides.pop("oauth_enabled", False)),
        allow_local_bypass=bool(overrides.pop("allow_local_bypass", False)),
        allowed_ips=tuple(overrides.pop("allowed_ips", ())),
    )


class TestAPIKeyAuthentication:
    def test_env_api_key_not_accepted_without_header(self, monkeypatch):
        monkeypatch.setenv("SIGIL_MCP_API_KEY", "env-secret")

        # Any value returned from verify_api_key should fail since no header present
        monkeypatch.setattr(security_auth, "verify_api_key", lambda key: False, raising=False)

        assert security_auth.check_authentication(
            request_headers={},
            client_ip="203.0.113.10",
            settings=_make_settings(),
        ) is False

    def test_header_api_key_matches_env_value(self, monkeypatch):
        monkeypatch.setenv("SIGIL_MCP_API_KEY", "env-secret")

        # Simulate missing API key file (verify_api_key should not be called)
        def _fail_verify(_key):  # pragma: no cover - defensive guard
            raise AssertionError("verify_api_key should not be called when env key matches header")

        monkeypatch.setattr(security_auth, "verify_api_key", _fail_verify, raising=False)

        assert security_auth.check_authentication(
            request_headers={"x-api-key": "env-secret"},
            client_ip="203.0.113.20",
            settings=_make_settings(),
        ) is True


class TestRedirectValidation:
    def test_allowed_redirect_in_allow_list(self):
        assert security_auth.is_redirect_uri_allowed(
            "https://chat.openai.com/aip/oauth/callback",
            registered_redirects=[],
            allow_list=["https://chat.openai.com"],
        ) is True

    def test_rejects_unlisted_redirect(self):
        assert security_auth.is_redirect_uri_allowed(
            "https://malicious.example.com/callback",
            registered_redirects=["https://trusted.example.com/callback"],
            allow_list=["https://chat.openai.com"],
        ) is False
