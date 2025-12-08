import sigil_mcp.server as server


def _configure_auth(monkeypatch):
    monkeypatch.setattr(server, "ALLOW_LOCAL_BYPASS", False, raising=False)
    monkeypatch.setattr(server, "AUTH_ENABLED", True, raising=False)
    monkeypatch.setattr(server, "OAUTH_ENABLED", False, raising=False)


class TestAPIKeyAuthentication:
    def test_env_api_key_not_accepted_without_header(self, monkeypatch):
        _configure_auth(monkeypatch)
        monkeypatch.setenv("SIGIL_MCP_API_KEY", "env-secret")

        # Any value returned from verify_api_key should fail since no header present
        monkeypatch.setattr(server, "verify_api_key", lambda key: False, raising=False)

        assert server.check_authentication(
            request_headers={},
            client_ip="203.0.113.10",
        ) is False

    def test_header_api_key_matches_env_value(self, monkeypatch):
        _configure_auth(monkeypatch)
        monkeypatch.setenv("SIGIL_MCP_API_KEY", "env-secret")

        # Simulate missing API key file (verify_api_key should not be called)
        def _fail_verify(_key):  # pragma: no cover - defensive guard
            raise AssertionError("verify_api_key should not be called when env key matches header")

        monkeypatch.setattr(server, "verify_api_key", _fail_verify, raising=False)

        assert server.check_authentication(
            request_headers={"x-api-key": "env-secret"},
            client_ip="203.0.113.20",
        ) is True


class TestRedirectValidation:
    def test_allowed_redirect_in_allow_list(self):
        assert server._is_redirect_uri_allowed(
            "https://chat.openai.com/aip/oauth/callback",
            registered_redirects=[],
            allow_list=["https://chat.openai.com"],
        ) is True

    def test_rejects_unlisted_redirect(self):
        assert server._is_redirect_uri_allowed(
            "https://malicious.example.com/callback",
            registered_redirects=["https://trusted.example.com/callback"],
            allow_list=["https://chat.openai.com"],
        ) is False
