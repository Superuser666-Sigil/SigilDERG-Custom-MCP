# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import copy

import pytest

from sigil_mcp import server


@pytest.fixture(autouse=True)
def restore_config():
    """Ensure mcp_server config mutations are restored between tests."""
    original = copy.deepcopy(server.config.config_data)
    yield
    server.config.config_data = original


@pytest.mark.anyio
async def test_sse_available_without_token():
    server.config.config_data.setdefault("mcp_server", {})["require_token"] = False
    app = server.build_parent_app(include_admin=False)
    base_app = app
    while hasattr(base_app, "app") and not hasattr(base_app, "routes"):
        base_app = base_app.app

    sse_mount = next(
        (r for r in base_app.routes if getattr(r, "path", None) == "/mcp"),
        None,
    )
    assert sse_mount is not None
    assert any(getattr(r, "path", "") == "/sse" for r in sse_mount.app.routes)


@pytest.mark.anyio
async def test_sse_requires_token_when_configured():
    mcp_cfg = server.config.config_data.setdefault("mcp_server", {})
    mcp_cfg["require_token"] = True
    mcp_cfg["token"] = "test-token"

    app = server.build_parent_app(include_admin=False)
    base_app = app
    while hasattr(base_app, "app") and not hasattr(base_app, "routes"):
        base_app = base_app.app

    sse_mount = next(
        (r for r in base_app.routes if getattr(r, "path", None) == "/mcp"),
        None,
    )
    assert sse_mount is not None
    middleware_classes = [m.cls for m in sse_mount.app.user_middleware]
    assert server.MCPBearerAuthMiddleware in middleware_classes
