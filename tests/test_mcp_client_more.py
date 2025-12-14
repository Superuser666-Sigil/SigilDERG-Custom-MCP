import asyncio
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from sigil_mcp.mcp_client import (
    ExternalMCPConfigError,
    ExternalMCPServer,
    MCPClientManager,
    MCPClientWrapper,
    _normalize_name,
    get_global_manager,
    set_global_manager,
)


def test_normalize_name_requires_value():
    with pytest.raises(ExternalMCPConfigError):
        _normalize_name("   ")


@pytest.mark.anyio
async def test_mcp_wrapper_connects_sse(monkeypatch):
    # Stub ClientSession to avoid hitting the real MCP transport
    created = {}

    class DummySession:
        def __init__(self, recv, send, read_timeout_seconds):
            created["args"] = (recv, send, read_timeout_seconds)

        async def list_tools(self):
            return SimpleNamespace(tools=[])

        async def call_tool(self, name, arguments):
            return SimpleNamespace(model_dump=lambda **_: {"name": name, **arguments})

    @asynccontextmanager
    async def fake_sse_client(url, headers=None, timeout=None, sse_read_timeout=None):
        async def send(msg=None):
            return None

        async def recv():
            return None

        yield send, recv

    monkeypatch.setattr("sigil_mcp.mcp_client.sse_client", fake_sse_client)
    monkeypatch.setattr("sigil_mcp.mcp_client.ClientSession", DummySession)

    server = ExternalMCPServer(name="srv", type="sse", url="http://example.com")
    wrapper = MCPClientWrapper(server, logger=logging.getLogger("test_mcp_wrapper"))

    tools = await wrapper.list_tools()
    assert tools.tools == []

    result = await wrapper.call_tool("echo", {"value": 1})
    assert result.model_dump() == {"name": "echo", "value": 1}
    assert created["args"][2].total_seconds() == pytest.approx(120.0)


def test_mcp_manager_skips_disabled_servers():
    mgr = MCPClientManager(
        [
            {"name": "a", "type": "stdio", "command": "echo", "disabled": True},
            {"name": "b", "type": "stdio", "command": "echo"},
        ]
    )
    assert all(server.name != "a" for server in mgr._servers)
    assert any(server.name == "b" for server in mgr._servers)
    client = mgr._build_client(mgr._servers[0])
    assert isinstance(client, MCPClientWrapper)


@pytest.mark.anyio
async def test_register_with_fastmcp_no_servers_returns_quickly():
    mgr = MCPClientManager([])
    await mgr.register_with_fastmcp(mcp_app=None)
    assert mgr._tools_by_server == {}


@pytest.mark.anyio
async def test_register_with_fastmcp_registers_helpers(monkeypatch):
    tool = SimpleNamespace(name="one", description="first", inputSchema=None)
    mgr = MCPClientManager([{"name": "srv", "type": "stdio", "command": "echo"}])
    mgr._tools_by_server = {"srv": [tool]}

    async def fake_discover():
        return None

    mgr.discover = fake_discover  # type: ignore[assignment]

    registered: dict[str, callable] = {}

    def decorator(**kwargs):
        def wrapper(fn):
            registered[kwargs["name"]] = fn
            return fn

        return wrapper

    await mgr.register_with_fastmcp(mcp_app=None, tool_decorator=decorator)

    assert "srv.one" in registered
    tools_payload = await registered["list_mcp_tools"]()
    assert tools_payload["tools"] == mgr.list_registered_tools()

    prompt_payload = await registered["external_mcp_prompt"]()
    assert "External MCP tools available" in prompt_payload["prompt"]


@pytest.mark.anyio
async def test_call_proxy_and_refresh(monkeypatch):
    class DummyClient:
        async def call_tool(self, name, arguments):
            return SimpleNamespace(model_dump=lambda **_: {"called": name, **arguments})

    mgr = MCPClientManager([])
    mgr._clients = {"srv": DummyClient()}

    result = await mgr._call_proxy("srv", "x", {"a": 1})
    assert result == {"called": "x", "a": 1}

    called = {}

    async def fake_register(mcp_app, tool_decorator=None):
        called["ran"] = True

    mgr.register_with_fastmcp = fake_register  # type: ignore[assignment]
    await mgr.refresh(mcp_app="app", tool_decorator=None)
    assert called.get("ran") is True


def test_status_and_close_and_global_manager(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True
            raise RuntimeError("boom")

    mgr = MCPClientManager([])
    mgr._clients = {"srv": DummyClient()}

    status = mgr.status()
    assert status["servers_configured"] == 0
    assert status["servers_active"] == 1

    asyncio.run(mgr.close())

    set_global_manager(mgr)
    assert get_global_manager() is mgr
    set_global_manager(None)
