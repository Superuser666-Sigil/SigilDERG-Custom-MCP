# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from sigil_mcp.mcp_client import ExternalMCPConfigError, MCPClientManager


class DummyClient:
    def __init__(self):
        self.called_with = None

    async def list_tools(self) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(
                    name="hello",
                    description="hello tool",
                    inputSchema={"type": "object", "properties": {"name": {"type": "string"}}},
                )
            ]
        )

    async def call_tool(self, tool_name, arguments):
        self.called_with = (tool_name, arguments)
        return CallToolResult(content=[TextContent(type="text", text="ok")])


class DummyMCP:
    def __init__(self):
        self.registered = {}

    def tool(self, **meta):
        def decorator(fn):
            self.registered[meta.get("name", fn.__name__)] = {
                "fn": fn,
                "meta": meta,
            }
            return fn

        return decorator


@pytest.mark.anyio
async def test_register_and_call_proxy(monkeypatch):
    cfg = [
        {"name": "Demo", "type": "stdio", "command": "echo"},
    ]
    manager = MCPClientManager(cfg)
    dummy_client = DummyClient()
    monkeypatch.setattr(manager, "_build_client", lambda _server: dummy_client)

    dummy_mcp = DummyMCP()
    await manager.register_with_fastmcp(dummy_mcp, tool_decorator=dummy_mcp.tool)

    assert "demo.hello" in dummy_mcp.registered
    # Call the registered wrapper to ensure it reaches the dummy client
    wrapper = dummy_mcp.registered["demo.hello"]["fn"]
    result = await wrapper(name="Sigil")
    assert result["content"][0]["text"] == "ok"
    assert dummy_client.called_with == ("hello", {"name": "Sigil"})

    # Diagnostics tools present
    assert "list_mcp_tools" in dummy_mcp.registered
    assert "external_mcp_prompt" in dummy_mcp.registered


def test_config_validation_duplicates():
    cfg = [
        {"name": "dup", "type": "stdio", "command": "python"},
        {"name": "dup", "type": "sse", "url": "http://example.com"},
    ]
    with pytest.raises(ExternalMCPConfigError):
        MCPClientManager(cfg)
