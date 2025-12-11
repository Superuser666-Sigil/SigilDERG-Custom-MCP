from sigil_mcp.security.auth import AuthSettings, check_authentication


def test_ip_whitelist_enforced_before_local_bypass():
    settings = AuthSettings(
        auth_enabled=True,
        oauth_enabled=False,
        allow_local_bypass=True,
        allowed_ips=("10.0.0.1",),
        mode="prod",
    )

    assert check_authentication(client_ip="127.0.0.1", settings=settings) is False
