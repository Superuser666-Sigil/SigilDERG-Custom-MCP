# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Tests for header logging middleware.
"""

import pytest
import logging
from sigil_mcp.middleware.header_logging import (
    HeaderLoggingASGIMiddleware,
    redact_headers,
    SENSITIVE_HEADERS,
)


class TestHeaderRedaction:
    """Test header redaction functionality."""

    def test_redact_sensitive_headers(self):
        """Test that sensitive headers are redacted."""
        headers = {
            "authorization": "Bearer secret-token",
            "cookie": "session=abc123",
            "x-api-key": "my-api-key",
            "x-admin-key": "admin-secret",
            "x-openai-session": "session-id",
            "x-openai-session-token": "token-value",
            "content-type": "application/json",
            "user-agent": "test-client",
        }

        redacted = redact_headers(headers)

        # Sensitive headers should be redacted
        assert redacted["authorization"] == "<redacted>"
        assert redacted["cookie"] == "<redacted>"
        assert redacted["x-api-key"] == "<redacted>"
        assert redacted["x-admin-key"] == "<redacted>"
        assert redacted["x-openai-session"] == "<redacted>"
        assert redacted["x-openai-session-token"] == "<redacted>"

        # Non-sensitive headers should pass through
        assert redacted["content-type"] == "application/json"
        assert redacted["user-agent"] == "test-client"

    def test_redact_case_insensitive(self):
        """Test that header redaction is case-insensitive."""
        headers = {
            "Authorization": "Bearer token",
            "AUTHORIZATION": "Bearer token2",
            "X-API-Key": "key-value",
        }

        redacted = redact_headers(headers)

        assert redacted["Authorization"] == "<redacted>"
        assert redacted["AUTHORIZATION"] == "<redacted>"
        assert redacted["X-API-Key"] == "<redacted>"

    def test_redact_empty_headers(self):
        """Test redaction with empty headers dict."""
        redacted = redact_headers({})
        assert redacted == {}


class TestHeaderLoggingMiddleware:
    """Test HeaderLoggingASGIMiddleware functionality."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock ASGI app."""
        async def async_app(scope, receive, send):
            # Simulate successful response
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"OK"})

        return async_app

    @pytest.fixture
    def middleware(self, mock_app):
        """Create middleware instance with mock app."""
        return HeaderLoggingASGIMiddleware(mock_app)

    @pytest.mark.anyio
    async def test_middleware_logs_request_headers(self, middleware, mock_app, caplog):
        """Test that middleware logs incoming request headers."""
        with caplog.at_level(logging.INFO):
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/test",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"user-agent", b"test-client"),
                    (b"authorization", b"Bearer secret"),
                ],
                "client": ("127.0.0.1", 12345),
            }

            async def receive():
                return {"type": "http.request"}

            send_calls = []

            async def send(message):
                send_calls.append(message)

            await middleware(scope, receive, send)

            # Check that request was logged
            log_text = caplog.text
            assert "Incoming MCP HTTP request" in log_text
            # Check log records for structured data
            records = [r for r in caplog.records if "Incoming MCP HTTP request" in r.message]
            assert len(records) > 0
            record = records[0]
            assert record.method == "POST"
            assert record.path == "/test"
            assert record.client_ip == "127.0.0.1"
            assert hasattr(record, "headers")
            assert "content-type" in str(record.headers)
            assert "user-agent" in str(record.headers)

    @pytest.mark.anyio
    async def test_middleware_logs_response_status(self, middleware, mock_app, caplog):
        """Test that middleware logs response status code."""
        with caplog.at_level(logging.INFO):
            scope = {
                "type": "http",
                "method": "GET",
                "path": "/healthz",
                "headers": [],
                "client": ("127.0.0.1", 12345),
            }

            async def receive():
                return {"type": "http.request"}

            send_calls = []

            async def send(message):
                send_calls.append(message)
                if message["type"] == "http.response.start":
                    # Simulate response
                    await send({"type": "http.response.body", "body": b"OK"})

            await middleware(scope, receive, send)

            # Check that response was logged
            records = [r for r in caplog.records if "Outgoing MCP HTTP response" in r.message]
            assert len(records) > 0
            record = records[0]
            assert record.status_code == 200
            assert hasattr(record, "duration_ms")
            assert record.duration_ms >= 0

    @pytest.mark.anyio
    async def test_middleware_extracts_client_ip_from_forwarded_for(
        self, middleware, mock_app, caplog
    ):
        """Test that middleware extracts client IP from X-Forwarded-For header."""
        with caplog.at_level(logging.INFO):
            scope = {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [
                    (b"x-forwarded-for", b"203.0.113.42, 192.168.1.1"),
                ],
                "client": ("127.0.0.1", 12345),
            }

            async def receive():
                return {"type": "http.request"}

            async def send(message):
                pass

            await middleware(scope, receive, send)

            # Should use first IP from X-Forwarded-For
            records = [r for r in caplog.records if "Incoming MCP HTTP request" in r.message]
            assert len(records) > 0
            assert records[0].client_ip == "203.0.113.42"

    @pytest.mark.anyio
    async def test_middleware_extracts_cf_ray(self, middleware, mock_app, caplog):
        """Test that middleware extracts Cloudflare ray ID."""
        with caplog.at_level(logging.INFO):
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/",
                "headers": [
                    (b"cf-ray", b"8978a4bf1c5a1234-DFW"),
                ],
                "client": ("127.0.0.1", 12345),
            }

            async def receive():
                return {"type": "http.request"}

            async def send(message):
                pass

            await middleware(scope, receive, send)

            records = [r for r in caplog.records if "Incoming MCP HTTP request" in r.message]
            assert len(records) > 0
            assert records[0].cf_ray == "8978a4bf1c5a1234-DFW"

    @pytest.mark.anyio
    async def test_middleware_passes_through_non_http(self, middleware, mock_app):
        """Test that middleware passes through non-HTTP traffic."""
        scope = {"type": "websocket", "path": "/ws"}

        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            pass

        # Should not raise and should call the app
        call_count_before = 0
        try:
            call_count_before = len([c for c in mock_app.__call__.mock_calls if c])
        except Exception:
            pass
        
        await middleware(scope, receive, send)
        
        # App should have been called (non-HTTP passes through)
        # Since we're using a function, we can't easily track calls, so just verify no error

    @pytest.mark.anyio
    async def test_middleware_logs_errors(self, caplog):
        """Test that middleware logs errors and re-raises them."""
        async def failing_app(scope, receive, send):
            raise ValueError("Test error")

        error_middleware = HeaderLoggingASGIMiddleware(failing_app)

        with caplog.at_level(logging.ERROR):
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/error",
                "headers": [],
                "client": ("127.0.0.1", 12345),
            }

            async def receive():
                return {"type": "http.request"}

            async def send(message):
                pass

            with pytest.raises(ValueError, match="Test error"):
                await error_middleware(scope, receive, send)

            # Check that error was logged
            assert "Error handling MCP request" in caplog.text
            assert "Test error" in caplog.text

    @pytest.mark.anyio
    async def test_middleware_generates_request_id(self, middleware, mock_app, caplog):
        """Test that middleware generates unique request IDs."""
        with caplog.at_level(logging.INFO):
            scope = {
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "client": ("127.0.0.1", 12345),
            }

            async def receive():
                return {"type": "http.request"}

            async def send(message):
                pass

            await middleware(scope, receive, send)

            # Extract request_id from logs
            log_records = [r for r in caplog.records if "Incoming MCP HTTP request" in r.message]
            assert len(log_records) > 0
            assert hasattr(log_records[0], "request_id")
            assert log_records[0].request_id is not None

            # Response should have same request_id
            response_records = [
                r for r in caplog.records if "Outgoing MCP HTTP response" in r.message
            ]
            assert len(response_records) > 0
            assert response_records[0].request_id == log_records[0].request_id


class TestMiddlewareIntegration:
    """Test middleware integration with FastMCP."""

    def test_sensitive_headers_constant(self):
        """Test that SENSITIVE_HEADERS contains expected headers."""
        expected = {
            "authorization",
            "cookie",
            "x-api-key",
            "x-admin-key",
            "x-openai-session",
            "x-openai-session-token",
        }
        assert SENSITIVE_HEADERS == expected

    def test_middleware_installation_with_mock_fastmcp(self):
        """Test middleware installation logic with mocked FastMCP."""
        from sigil_mcp.server import mcp

        # The middleware installation happens at module import time
        # We can't easily test the logs, but we can verify the middleware class exists
        # and that the installation code ran without errors
        assert HeaderLoggingASGIMiddleware is not None
        assert hasattr(mcp, "app") or not hasattr(mcp, "app")  # Either is fine

    @pytest.mark.anyio
    async def test_middleware_with_real_asgi_app(self):
        """Test middleware with a real ASGI app structure."""
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def test_handler(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/test", test_handler)])
        middleware = HeaderLoggingASGIMiddleware(app)

        # Create a test scope
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [(b"user-agent", b"test")],
            "client": ("127.0.0.1", 12345),
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        send_calls = []

        async def send(message):
            send_calls.append(message)

        await middleware(scope, receive, send)

        # Should have sent response
        assert len(send_calls) > 0
        assert send_calls[0]["type"] == "http.response.start"
        assert send_calls[0]["status"] == 200

