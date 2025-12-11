# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import pytest

import sigil_mcp.mcp_client as mcp_client


def test_normalize_name_sanitizes():
    assert mcp_client._normalize_name("Playwright-Prod") == "playwright_prod"
    with pytest.raises(mcp_client.ExternalMCPConfigError):
        mcp_client._normalize_server_type("bad")


def test_external_mcp_server_from_dict_stdio_requires_command():
    with pytest.raises(mcp_client.ExternalMCPConfigError):
        mcp_client.ExternalMCPServer.from_dict({"name": "stdio1", "type": "stdio"})


def test_external_mcp_server_from_dict_ok_sse():
    server = mcp_client.ExternalMCPServer.from_dict(
        {"name": "sse1", "type": "sse", "url": "http://x"}
    )
    assert server.type == "sse"
