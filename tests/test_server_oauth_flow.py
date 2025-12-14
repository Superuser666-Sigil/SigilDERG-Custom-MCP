# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import types
from urllib.parse import urlparse, parse_qs

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import sigil_mcp.server as server
from sigil_mcp import oauth as oauth_mod


@pytest.fixture
def oauth_setup(monkeypatch):
    """Reset OAuth files and manager so each test uses a fresh client."""
    monkeypatch.setattr(server, "OAUTH_ENABLED", True)
    # Clear persisted state in the temp OAuth dir
    for path in [oauth_mod.CLIENT_FILE, oauth_mod.TOKENS_FILE, oauth_mod.STATE_FILE]:
        path.unlink(missing_ok=True)
    oauth_mod._oauth_manager = None  # type: ignore[attr-defined]
    mgr = oauth_mod.get_oauth_manager()
    mgr.tokens.clear()
    mgr.codes.clear()
    mgr.states.clear()
    mgr.initialize_client()
    client = mgr.get_client()
    # Prefer a redirect URI that is already in the default allow list
    redirect_uri = next(
        (uri for uri in client.redirect_uris if "chatgpt.com" in uri),
        client.redirect_uris[0],
    )
    yield {"client_id": client.client_id, "client_secret": client.client_secret, "redirect_uri": redirect_uri, "manager": mgr}
    # Cleanup
    mgr.tokens.clear()
    mgr.codes.clear()
    mgr.states.clear()


def test_oauth_http_flow_end_to_end(oauth_setup, monkeypatch):
    monkeypatch.setattr(server, "OAUTH_ENABLED", True)
    # Ensure redirect allow list accepts the chosen URI
    monkeypatch.setattr(
        server,
        "config",
        types.SimpleNamespace(
            oauth_redirect_allow_list=[oauth_setup["redirect_uri"]],
            auth_enabled=False,
            allow_local_bypass=True,
            oauth_enabled=True,
            admin_enabled=False,
            mode="dev",
            mcp_sse_path="/sse",
            mcp_message_path="/message",
            mcp_http_path="/",
        ),
    )

    app = Starlette(
        routes=[Route("/oauth/authorize", server.oauth_authorize_http, methods=["GET", "POST"])]
    )
    client = TestClient(app)
    params = {
        "client_id": oauth_setup["client_id"],
        "redirect_uri": oauth_setup["redirect_uri"],
        "response_type": "code",
        "state": "xyz",
    }
    consent = client.get("/oauth/authorize", params=params)
    assert consent.status_code == 200
    # Approve via POST to get redirected with an authorization code
    redirect = client.post("/oauth/authorize", data={**params, "approve": "true"}, follow_redirects=False)
    assert redirect.status_code == 302
    loc = redirect.headers["location"]
    parsed = urlparse(loc)
    code = parse_qs(parsed.query)["code"][0]

    token_app = Starlette(routes=[Route("/oauth/token", server.oauth_token_http, methods=["POST"])])
    token_client = TestClient(token_app)
    token_resp = token_client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": oauth_setup["redirect_uri"],
            "client_id": oauth_setup["client_id"],
        },
    )
    assert token_resp.status_code == 200
    token_data = token_resp.json()
    assert token_data["access_token"]

    refresh_resp = token_client.post(
        "/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": token_data["refresh_token"],
            "client_id": oauth_setup["client_id"],
        },
    )
    assert refresh_resp.status_code == 200

    revoke_app = Starlette(routes=[Route("/oauth/revoke", server.oauth_revoke_http, methods=["POST"])])
    revoke_client = TestClient(revoke_app)
    revoke_resp = revoke_client.post(
        "/oauth/revoke",
        data={"token": token_data["access_token"], "client_id": oauth_setup["client_id"]},
    )
    assert revoke_resp.status_code == 200
    assert revoke_resp.json()["status"] in {"revoked", "not_found"}

    # Client info should now be available
    info = server.oauth_client_info()
    assert info["client_id"] == oauth_setup["client_id"]
    assert "/oauth/authorize" in info["authorization_endpoint"]


def test_oauth_tool_functions(monkeypatch, oauth_setup):
    monkeypatch.setattr(server, "OAUTH_ENABLED", True)
    # Allow redirects for the chosen URI
    monkeypatch.setattr(
        server,
        "config",
        types.SimpleNamespace(
            oauth_redirect_allow_list=[oauth_setup["redirect_uri"]],
            auth_enabled=False,
            allow_local_bypass=True,
            oauth_enabled=True,
            admin_enabled=False,
            mode="dev",
        ),
    )

    auth_resp = server.oauth_authorize(
        client_id=oauth_setup["client_id"],
        redirect_uri=oauth_setup["redirect_uri"],
        state="abc",
    )
    assert "code" in auth_resp
    code = auth_resp["code"]

    token_resp = server.oauth_token(
        grant_type="authorization_code",
        code=code,
        redirect_uri=oauth_setup["redirect_uri"],
        client_id=oauth_setup["client_id"],
    )
    assert "access_token" in token_resp
    assert token_resp["refresh_token"]

    refresh_resp = server.oauth_token(
        grant_type="refresh_token",
        refresh_token=token_resp["refresh_token"],
        client_id=oauth_setup["client_id"],
    )
    assert refresh_resp["access_token"]

    revoke_resp = server.oauth_revoke(
        token=token_resp["access_token"],
        client_id=oauth_setup["client_id"],
    )
    assert revoke_resp["status"] in {"revoked", "not_found"}

    # Unsupported grant type and disabled flag branches
    bad_grant = server.oauth_token(grant_type="bogus", client_id=oauth_setup["client_id"])
    assert bad_grant["error"] == "unsupported_grant_type"

    monkeypatch.setattr(server, "OAUTH_ENABLED", False)
    disabled = server.oauth_authorize(
        client_id=oauth_setup["client_id"],
        redirect_uri=oauth_setup["redirect_uri"],
    )
    assert disabled["error"] == "oauth_not_enabled"
    disabled_info = server.oauth_client_info()
    assert disabled_info["error"] == "oauth_not_enabled"
