# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

from __future__ import annotations

import logging
import os

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .config import Config, get_config
from .logging_setup import get_log_file_path, setup_logging
from .middleware.header_logging import HeaderLoggingASGIMiddleware


logger = logging.getLogger("sigil_repos_mcp")


def _configure_logging(config: Config) -> None:
    # Skip file logging in test environments to avoid permission issues
    if os.getenv("PYTEST_CURRENT_TEST"):
        log_file_path = None
    else:
        log_file_path = get_log_file_path(config.log_file)
    try:
        setup_logging(
            log_file=str(log_file_path) if log_file_path else None,
            log_level=config.log_level,
            console_output=True,
        )
        if log_file_path:
            logger.info("Logging to file: %s", log_file_path)
        else:
            logger.info("Logging to console only (no log file configured)")
    except Exception as exc:
        logger.warning("Falling back to console logging (file logging failed): %s", exc)
        setup_logging(log_file=None, log_level=config.log_level, console_output=True)


class ChatGPTComplianceMiddleware:
    """
    Normalizes ChatGPT's non-compliant requests to standard JSON-RPC.
    Handles: Content-Type: application/octet-stream -> application/json
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST":
            headers = dict(scope.get("headers", []))
            ct = headers.get(b"content-type", b"")
            if b"application/octet-stream" in ct:
                new_headers = []
                for key, value in scope["headers"]:
                    if key.lower() != b"content-type":
                        new_headers.append((key, value))
                    else:
                        new_headers.append((b"content-type", b"application/json"))
                #
                scope["headers"] = new_headers

        await self.app(scope, receive, send)


def _wrap_for_chatgpt(mcp_server: FastMCP) -> None:
    underlying_app = getattr(mcp_server, "app", None)
    if underlying_app is not None and not isinstance(underlying_app, ChatGPTComplianceMiddleware):
        mcp_server.app = ChatGPTComplianceMiddleware(underlying_app)  # type: ignore[attr-defined]
        return

    underlying_asgi_app = getattr(mcp_server, "asgi_app", None)
    if underlying_asgi_app is not None and not isinstance(
        underlying_asgi_app, ChatGPTComplianceMiddleware
    ):
        setattr(
            mcp_server,
            "asgi_app",
            ChatGPTComplianceMiddleware(underlying_asgi_app),
        )


def build_mcp_app(
    config: Config | None = None,
    *,
    enable_chatgpt_compliance: bool | None = None,
    enable_header_logging: bool | None = None,
) -> FastMCP:
    """Construct the FastMCP app with optional middleware toggles."""

    config = config or get_config()
    _configure_logging(config)

    apply_chatgpt_mw = (
        config.chatgpt_compliance_enabled
        if enable_chatgpt_compliance is None
        else enable_chatgpt_compliance
    )
    apply_header_logging = (
        config.header_logging_enabled
        if enable_header_logging is None
        else enable_header_logging
    )

    transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

    mcp = FastMCP(
        name=config.server_name,
        json_response=True,
        streamable_http_path="/",
        transport_security=transport_security,
    )

    if apply_chatgpt_mw:
        _wrap_for_chatgpt(mcp)
    else:
        logger.info("ChatGPTComplianceMiddleware disabled by configuration")

    if apply_header_logging and not config.admin_enabled:
        underlying_app = getattr(mcp, "app", None) or getattr(mcp, "asgi_app", None)
        if underlying_app is not None:
            wrapped_app = HeaderLoggingASGIMiddleware(underlying_app)
            if hasattr(mcp, "app"):
                mcp.app = wrapped_app  # type: ignore[attr-defined]
            elif hasattr(mcp, "asgi_app"):
                mcp.asgi_app = wrapped_app  # type: ignore[attr-defined]
            logger.info("HeaderLoggingASGIMiddleware installed on FastMCP app")
        else:
            logger.warning(
                "Could not locate underlying ASGI app on FastMCP; "
                "header logging middleware not installed. "
                "Check FastMCP API for correct attribute name."
            )
    else:
        logger.info(
            "Header logging middleware disabled or admin API enabled - skipping"
        )

    return mcp
