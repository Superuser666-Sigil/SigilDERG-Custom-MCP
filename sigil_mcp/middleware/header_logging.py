# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

from __future__ import annotations

import logging
import time
import uuid

logger = logging.getLogger("sigil_repos_mcp")

SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "x-api-key",
    "x-admin-key",
    "x-openai-session",
    "x-openai-session-token",
}


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of headers with sensitive keys redacted."""

    redacted: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in SENSITIVE_HEADERS:
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def redact_querystring(path: str) -> str:
    """Redact query parameters that may contain secrets."""
    if "?" not in path:
        return path
    base, qs = path.split("?", 1)
    # Drop all query params to avoid leaking tokens in logs
    return f"{base}?<redacted>"


class HeaderLoggingASGIMiddleware:
    """
    ASGI middleware that logs incoming request headers and outgoing status codes.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        request_id = str(uuid.uuid4())

        headers_dict: dict[str, str] = {}
        for raw_name, raw_value in scope.get("headers", []):
            name = raw_name.decode("latin-1")
            value = raw_value.decode("latin-1")
            headers_dict[name] = value

        path = scope.get("path", "")
        if scope.get("query_string"):
            try:
                qs = scope["query_string"].decode("latin-1")
                path = redact_querystring(f"{path}?{qs}")
            except Exception:
                path = redact_querystring(path)
        method = scope.get("method", "UNKNOWN")

        client_ip = headers_dict.get("x-forwarded-for")
        if client_ip:
            client_ip = client_ip.split(",")[0].strip()
        else:
            client = scope.get("client")
            if client and isinstance(client, (tuple, list)) and len(client) >= 1:
                client_ip = client[0]
            else:
                client_ip = None

        cf_ray = headers_dict.get("cf-ray")
        safe_headers = redact_headers(headers_dict)

        logger.info(
            "Incoming MCP HTTP request",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_ip": client_ip,
                "cf_ray": cf_ray,
                "headers": safe_headers,
            },
        )

        status_code_holder = {"value": None}
        response_body_parts: list[bytes] = []
        capture_body = True
        max_capture_bytes = 4096
        captured_bytes = 0
        truncated = False

        async def send_wrapper(message):
            nonlocal capture_body, captured_bytes, truncated
            if message["type"] == "http.response.start":
                status_code_holder["value"] = message.get("status", None)
                headers = list(message.get("headers", []))
                headers.append((b"x-accel-buffering", b"no"))
                headers.append((b"cache-control", b"no-cache, no-store, must-revalidate"))
                headers.append((b"connection", b"keep-alive"))
                message["headers"] = headers
            elif message["type"] == "http.response.body":
                if capture_body:
                    body = message.get("body", b"")
                    if message.get("more_body"):
                        capture_body = False
                        truncated = True  # treat streaming bodies as truncated for logging
                    if body and not truncated:
                        remaining = max_capture_bytes - captured_bytes
                        if remaining > 0:
                            response_body_parts.append(body[:remaining])
                            captured_bytes += min(len(body), remaining)
                        if captured_bytes >= max_capture_bytes:
                            truncated = True
                            capture_body = False
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.exception(
                "Error handling MCP request",
                extra={
                    "request_id": request_id,
                    "duration_ms": duration_ms,
                    "path": path,
                    "method": method,
                },
            )
            raise

        duration_ms = int((time.time() - start_time) * 1000)
        response_body = b"".join(response_body_parts).decode("utf-8", errors="replace")
        if response_body:
            if len(response_body) > 500:
                response_body = response_body[:500] + "... [truncated]"
            elif truncated:
                response_body = response_body + "... [truncated]"

        log_extra = {
            "request_id": request_id,
            "status_code": status_code_holder["value"],
            "duration_ms": duration_ms,
            "path": path,
            "method": method,
        }
        if path == "/" and response_body:
            log_extra["response_body"] = response_body

        logger.info("Outgoing MCP HTTP response", extra=log_extra)
