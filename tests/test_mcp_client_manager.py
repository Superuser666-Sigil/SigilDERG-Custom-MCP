# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import asyncio

from mcp.types import CallToolResult, ListToolsResult, Tool

from sigil_mcp.mcp_client import MCPClientManager


class FakeClient:
    def __init__(self, tools):
        self._tools = tools
        self.calls = []

    async def list_tools(self):
        return ListToolsResult(tools=self._tools)

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return CallToolResult(content=[], isError=False)


async def _register_and_prompt(manager):
    await manager.register_with_fastmcp(SimpleMCP(), tool_decorator=lambda **_: (lambda fn: fn))
    return manager.prompt_snippet()


class SimpleMCP:
    def __init__(self):
        self.tools = []


def test_mcp_manager_registers_and_prompts(monkeypatch):
    tools = [Tool(name="t1", description="d1", inputSchema={})]
    fake = FakeClient(tools)
    mgr = MCPClientManager([{"name": "s1", "type": "sse", "url": "http://x"}])
    monkeypatch.setattr(mgr, "_build_client", lambda server: fake)

    asyncio.run(mgr.discover())
    assert mgr.list_registered_tools()[0]["name"] == "t1"

    snippet = asyncio.run(_register_and_prompt(mgr))
    assert "s1.t1" in snippet
