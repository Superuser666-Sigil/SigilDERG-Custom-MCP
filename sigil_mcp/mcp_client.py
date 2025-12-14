# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
External MCP client aggregation for Sigil.

Allows Sigil to connect to remote/local MCP servers (stdio, SSE, streamable HTTP),
discover their tools, and expose them via Sigil's FastMCP instance with server-prefixed
names (e.g., ``playwright.click``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, ListToolsResult, Tool


class ExternalMCPConfigError(Exception):
    """Raised when external MCP server configuration is invalid."""


def _normalize_server_type(raw: str) -> str:
    val = (raw or "").strip().lower()
    if val in {"stdio"}:
        return "stdio"
    if val in {"sse"}:
        return "sse"
    if val in {"http", "http-stream", "streaming-http", "streamable-http", "http-streaming"}:
        return "streamable-http"
    raise ExternalMCPConfigError(f"Unsupported MCP server type '{raw}'")


def _normalize_name(raw: str) -> str:
    name = (raw or "").strip().lower()
    if not name:
        raise ExternalMCPConfigError("MCP server name is required")
    safe = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            safe.append(ch)
        elif ch in {"-", " "}:
            safe.append("_")
        else:
            safe.append("_")
    return "".join(safe)


@dataclass
class ExternalMCPServer:
    name: str
    type: str
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, Any] = field(default_factory=dict)
    cwd: str | None = None
    encoding: str = "utf-8"
    encoding_error_handler: str = "strict"
    init_timeout: float = 10.0
    tool_timeout: float = 120.0
    verify: bool = True
    disabled: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalMCPServer:
        name = _normalize_name(data.get("name", ""))
        type_norm = _normalize_server_type(data.get("type", ""))

        if type_norm == "stdio" and not data.get("command"):
            raise ExternalMCPConfigError(f"stdio server '{name}' requires 'command'")
        if type_norm != "stdio" and not data.get("url"):
            raise ExternalMCPConfigError(f"remote server '{name}' requires 'url'")

        return cls(
            name=name,
            type=type_norm,
            url=data.get("url"),
            command=data.get("command"),
            args=list(data.get("args", [])),
            env=dict(data.get("env", {})),
            headers=dict(data.get("headers", {})),
            cwd=data.get("cwd"),
            encoding=data.get("encoding", "utf-8"),
            encoding_error_handler=data.get("encoding_error_handler", "strict"),
            init_timeout=float(data.get("init_timeout", 10.0)),
            tool_timeout=float(data.get("tool_timeout", 120.0)),
            verify=bool(data.get("verify", True)),
            disabled=bool(data.get("disabled", False)),
        )


class MCPClientWrapper:
    """Manages a single MCP client connection and session."""

    def __init__(self, server: ExternalMCPServer, logger: logging.Logger):
        self.server = server
        self.logger = logger
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()

    async def _connect(self) -> None:
        if self._session is not None:
            return

        self._stack = AsyncExitStack()
        try:
            if self.server.type == "stdio":
                params = StdioServerParameters(
                    command=self.server.command or "",
                    args=self.server.args,
                    env=self.server.env or None,
                    cwd=self.server.cwd,
                    encoding=self.server.encoding,
                    encoding_error_handler=self.server.encoding_error_handler,
                )
                send, recv = await self._stack.enter_async_context(stdio_client(params))
            elif self.server.type == "sse":
                send, recv = await self._stack.enter_async_context(
                    sse_client(
                        self.server.url or "",
                        headers=self.server.headers or None,
                        timeout=self.server.init_timeout,
                        sse_read_timeout=self.server.tool_timeout,
                    )
                )
            else:
                # streamable-http
                send, recv, _session_id = await self._stack.enter_async_context(
                    streamablehttp_client(
                        self.server.url or "",
                        headers=self.server.headers or None,
                        timeout=self.server.init_timeout,
                        sse_read_timeout=self.server.tool_timeout,
                    )
                )

            self._session = ClientSession(
                recv,
                send,
                read_timeout_seconds=timedelta(seconds=self.server.tool_timeout),
            )
        except Exception:
            if self._stack:
                await self._stack.aclose()
            self._stack = None
            self._session = None
            raise

    async def list_tools(self) -> ListToolsResult:
        async with self._lock:
            await self._connect()
            assert self._session is not None
            return await self._session.list_tools()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        async with self._lock:
            await self._connect()
            assert self._session is not None
            return await self._session.call_tool(tool_name, arguments)

    async def close(self) -> None:
        async with self._lock:
            if self._stack:
                await self._stack.aclose()
            self._stack = None
            self._session = None


class MCPClientManager:
    """Aggregates multiple external MCP servers and registers their tools with FastMCP."""

    def __init__(self, servers: Iterable[dict[str, Any]], logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("sigil_repos_mcp")
        self._servers: list[ExternalMCPServer] = []
        self._clients: dict[str, MCPClientWrapper] = {}
        self._tools_by_server: dict[str, list[Tool]] = {}
        self._last_refresh_error: str | None = None
        self._validate_and_set_servers(list(servers))

    def _validate_and_set_servers(self, servers: list[dict[str, Any]]) -> None:
        seen = set()
        for raw in servers:
            server = ExternalMCPServer.from_dict(raw)
            if server.disabled:
                continue
            if server.name in seen:
                raise ExternalMCPConfigError(f"Duplicate MCP server name '{server.name}'")
            seen.add(server.name)
            self._servers.append(server)

    def _build_client(self, server: ExternalMCPServer) -> MCPClientWrapper:
        return MCPClientWrapper(server, self.logger)

    async def discover(self) -> None:
        """Connect to all configured servers and cache their tools."""
        self._tools_by_server.clear()
        self._clients.clear()
        self._last_refresh_error = None

        for server in self._servers:
            client = self._build_client(server)
            try:
                tools_result = await client.list_tools()
                self._clients[server.name] = client
                self._tools_by_server[server.name] = list(tools_result.tools)
                self.logger.info(
                    "MCP external server '%s' available with %d tools",
                    server.name,
                    len(tools_result.tools),
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to initialize MCP server '%s': %s", server.name, exc
                )
                self._last_refresh_error = f"{server.name}: {exc}"

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        client = self._clients.get(server_name)
        if not client:
            raise RuntimeError(f"MCP server '{server_name}' is not available")
        return await client.call_tool(tool_name, arguments)

    def list_registered_tools(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for server, tools in self._tools_by_server.items():
            for tool in tools:
                items.append(
                    {
                        "server": server,
                        "name": tool.name,
                        "description": tool.description,
                    }
                )
        return items

    def prompt_snippet(self) -> str:
        """Return a human-readable prompt snippet describing external tools."""
        lines = ["External MCP tools available:"]
        for server, tools in self._tools_by_server.items():
            lines.append(f"- Server '{server}':")
            for tool in tools:
                desc = tool.description or ""
                lines.append(f"  - {server}.{tool.name}: {desc}".strip())
        if len(lines) == 1:
            return "No external MCP tools registered."
        return "\n".join(lines)

    async def register_with_fastmcp(self, mcp_app, *, tool_decorator: Callable[..., Callable] | None = None) -> None:
        """
        Discover tools and register them onto the provided FastMCP instance.
        """
        if not self._servers:
            return

        await self.discover()
        dec_factory = tool_decorator or (lambda **_: (lambda fn: fn))

        for server_name, tools in self._tools_by_server.items():
            for tool in tools:
                full_name = f"{server_name}.{tool.name}"
                decorator = dec_factory(
                    name=full_name,
                    title=tool.name,
                    description=tool.description or f"External MCP tool '{tool.name}' from '{server_name}'",
                    inputSchema=getattr(tool, "inputSchema", None),
                )

                async def _wrapper(_server: str = server_name, _tool: str = tool.name, **kwargs: Any) -> dict[str, Any]:
                    result = await self.call_tool(_server, _tool, kwargs)
                    return result.model_dump(exclude_none=True)

                decorated = decorator(_wrapper)
                decorated  # noqa: B018 (intentional no-op to satisfy linter)

        # Add diagnostics tool
        dec_diag = dec_factory(
            name="list_mcp_tools",
            title="List external MCP tools",
            description="List external MCP tools aggregated into Sigil",
        )

        async def list_mcp_tools() -> dict[str, Any]:
            return {"tools": self.list_registered_tools()}

        dec_diag(list_mcp_tools)

        # Prompt helper tool
        dec_prompt = dec_factory(
            name="external_mcp_prompt",
            title="External MCP prompt helper",
            description="Returns a prompt snippet describing external MCP tools registered in Sigil",
        )

        async def external_mcp_prompt() -> dict[str, Any]:
            return {"prompt": self.prompt_snippet()}

        dec_prompt(external_mcp_prompt)

    async def _call_proxy(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.call_tool(server_name, tool_name, arguments)
        return result.model_dump(exclude_none=True)

    async def refresh(self, mcp_app, *, tool_decorator: Callable[..., Callable] | None = None) -> None:
        """Reconnect and re-register tools (used by admin refresh endpoint)."""
        await self.register_with_fastmcp(mcp_app, tool_decorator=tool_decorator)

    def status(self) -> dict[str, Any]:
        return {
            "servers_configured": len(self._servers),
            "servers_active": len(self._clients),
            "last_error": self._last_refresh_error,
            "tools": self.list_registered_tools(),
        }

    async def close(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                continue


_GLOBAL_MANAGER: MCPClientManager | None = None


def set_global_manager(manager: MCPClientManager) -> None:
    global _GLOBAL_MANAGER
    _GLOBAL_MANAGER = manager


def get_global_manager() -> MCPClientManager | None:
    return _GLOBAL_MANAGER
