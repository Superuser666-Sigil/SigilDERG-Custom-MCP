# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import logging
import types

import pytest

from sigil_mcp.mcp_client import (
    ExternalMCPConfigError,
    ExternalMCPServer,
    MCPClientManager,
    MCPClientWrapper,
    _normalize_name,
    _normalize_server_type,
)


def test_normalize_server_type_invalid():
    with pytest.raises(ExternalMCPConfigError):
        _normalize_server_type("unknown")


def test_external_mcp_server_validation_errors():
    with pytest.raises(ExternalMCPConfigError):
        ExternalMCPServer.from_dict({"name": "stdio-missing", "type": "stdio"})

    with pytest.raises(ExternalMCPConfigError):
        ExternalMCPServer.from_dict({"name": "remote-missing", "type": "sse"})


def test_mcp_manager_duplicate_names():
    servers = [
        {"name": "dup", "type": "stdio", "command": "echo"},
        {"name": "dup", "type": "stdio", "command": "echo"},
    ]
    with pytest.raises(ExternalMCPConfigError):
        MCPClientManager(servers)


@pytest.mark.anyio
async def test_mcp_manager_discover_handles_errors(monkeypatch):
    server_cfg = {"name": "svc", "type": "stdio", "command": "echo"}
    manager = MCPClientManager([server_cfg])

    class FakeClient:
        async def list_tools(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(manager, "_build_client", lambda server: FakeClient())
    await manager.discover()
    assert manager.list_registered_tools() == []
    assert "svc" in (manager._last_refresh_error or "")


@pytest.mark.anyio
async def test_mcp_manager_prompt_snippet_and_call_tool(monkeypatch):
    manager = MCPClientManager([{"name": "svc", "type": "stdio", "command": "echo"}])
    assert manager.prompt_snippet() == "No external MCP tools registered."
    with pytest.raises(RuntimeError):
        await manager.call_tool("svc", "tool", {})

    tool_obj = types.SimpleNamespace(name="t1", description="desc")
    manager._tools_by_server = {"svc": [tool_obj]}
    snippet = manager.prompt_snippet()
    assert "t1" in snippet


def test_normalize_server_type_aliases():
    assert _normalize_server_type("STREAMING-HTTP") == "streamable-http"
    assert _normalize_server_type("http-stream") == "streamable-http"
    with pytest.raises(ExternalMCPConfigError):
        _normalize_server_type("")


def test_normalize_name_sanitizes_characters():
    name = _normalize_name(" My-Server! ")
    assert name == "my_server_"


@pytest.mark.anyio
async def test_mcp_client_wrapper_connects(monkeypatch):
    server_cfg = ExternalMCPServer.from_dict({"name": "svc", "type": "streamable-http", "url": "http://localhost"})
    wrapper = MCPClientWrapper(server_cfg, logger=logging.getLogger("test"))

    class DummyContext:
        async def __aenter__(self):
            return ("send", "recv", "session")

        async def __aexit__(self, *a):
            return False

    class DummySession:
        def __init__(self, recv, send, read_timeout_seconds):
            self.recv = recv
            self.send = send
            self.read_timeout_seconds = read_timeout_seconds

        async def list_tools(self):
            return types.SimpleNamespace(tools=[])

        async def call_tool(self, tool, args):
            return {"ok": True, "tool": tool, "args": args}

    monkeypatch.setattr("sigil_mcp.mcp_client.streamablehttp_client", lambda *a, **k: DummyContext())
    monkeypatch.setattr("sigil_mcp.mcp_client.ClientSession", DummySession)

    tools = await wrapper.list_tools()
    assert tools.tools == []
    res = await wrapper.call_tool("ping", {})
    assert res["tool"] == "ping"
    await wrapper.close()


@pytest.mark.anyio
async def test_mcp_client_wrapper_handles_connect_error(monkeypatch):
    server_cfg = ExternalMCPServer.from_dict({"name": "svc", "type": "stdio", "command": "echo"})
    wrapper = MCPClientWrapper(server_cfg, logger=logging.getLogger("test"))

    class FailingContext:
        async def __aenter__(self):
            raise RuntimeError("boom")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr("sigil_mcp.mcp_client.stdio_client", lambda params: FailingContext())
    with pytest.raises(RuntimeError):
        await wrapper.list_tools()
