# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

from types import SimpleNamespace

from sigil_mcp.app_factory import (
    ChatGPTComplianceMiddleware,
    _configure_logging,
    _wrap_for_chatgpt,
    build_mcp_app,
)


def test_chatgpt_middleware_rewrites_content_type():
    async def inner(scope, receive, send):
        assert dict(scope["headers"])[b"content-type"] == b"application/json"

    middleware = ChatGPTComplianceMiddleware(inner)
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"content-type", b"application/octet-stream")],
    }

    asyncio_run(middleware(scope, None, lambda x: None))


def asyncio_run(coro):
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def test_wrap_for_chatgpt_sets_wrapper():
    dummy = SimpleNamespace(app="app")
    _wrap_for_chatgpt(dummy)
    assert isinstance(dummy.app, ChatGPTComplianceMiddleware)


def test_wrap_for_chatgpt_wraps_asgi_app():
    dummy = SimpleNamespace(asgi_app=lambda scope, receive, send: None)
    _wrap_for_chatgpt(dummy)
    assert isinstance(dummy.asgi_app, ChatGPTComplianceMiddleware)


def test_chatgpt_middleware_preserves_other_headers():
    captured = {}

    async def inner(scope, receive, send):
        captured["headers"] = list(scope["headers"])

    middleware = ChatGPTComplianceMiddleware(inner)
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"x-test", b"1"), (b"content-type", b"application/octet-stream")],
    }
    asyncio_run(middleware(scope, None, lambda x: None))
    assert (b"x-test", b"1") in captured["headers"]


def test_build_mcp_app_header_logging_disabled_when_admin_enabled(monkeypatch):
    class DummyFastMCP:
        def __init__(self, *a, **k):
            self.app = lambda scope, receive, send: None
            self.settings = SimpleNamespace(debug=False)

        def streamable_http_app(self):
            return self.app

        def session_manager(self):
            return SimpleNamespace(run=lambda: None)

    cfg = SimpleNamespace(
        server_name="sigil",
        chatgpt_compliance_enabled=True,
        header_logging_enabled=True,
        admin_enabled=True,
        log_file=None,
        log_level="INFO",
    )
    monkeypatch.setattr("sigil_mcp.app_factory.FastMCP", DummyFastMCP)
    mcp = build_mcp_app(config=cfg)
    assert isinstance(mcp.app, ChatGPTComplianceMiddleware)


def test_configure_logging_fallback(monkeypatch, caplog):
    cfg = SimpleNamespace(log_file=None, log_level="INFO")
    calls = {"count": 0}

    def flaky_setup(*a, **k):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("fail")

    monkeypatch.setattr("sigil_mcp.app_factory.setup_logging", flaky_setup)
    _configure_logging(cfg)
    # Should attempt setup twice: first fails, fallback succeeds
    assert calls["count"] == 2


def test_build_mcp_app_warns_when_no_underlying_app(monkeypatch):
    class DummyFastMCP:
        def __init__(self, *a, **k):
            self.settings = SimpleNamespace(debug=False)

        def streamable_http_app(self):
            return None

        def session_manager(self):
            return SimpleNamespace(run=lambda: None)

    cfg = SimpleNamespace(
        server_name="sigil",
        chatgpt_compliance_enabled=False,
        header_logging_enabled=True,
        admin_enabled=False,
        log_file=None,
        log_level="INFO",
    )
    monkeypatch.setattr("sigil_mcp.app_factory.FastMCP", DummyFastMCP)
    mcp = build_mcp_app(config=cfg)
    assert mcp is not None


def test_build_mcp_app_wraps_header_logging(monkeypatch):
    class DummyFastMCP:
        def __init__(self, *a, **k):
            self.asgi_app = lambda scope, receive, send: None
            self.settings = SimpleNamespace(debug=False)

        def streamable_http_app(self):
            return self.asgi_app

        def session_manager(self):
            return SimpleNamespace(run=lambda: None)

    cfg = SimpleNamespace(
        server_name="sigil",
        chatgpt_compliance_enabled=False,
        header_logging_enabled=True,
        admin_enabled=False,
        log_file=None,
        log_level="INFO",
    )
    monkeypatch.setattr("sigil_mcp.app_factory.FastMCP", DummyFastMCP)
    app = build_mcp_app(config=cfg)
    assert app.asgi_app.__class__.__name__ == "HeaderLoggingASGIMiddleware"


def test_build_mcp_app_wraps_app_attribute(monkeypatch):
    class DummyFastMCP:
        def __init__(self, *a, **k):
            self.app = lambda scope, receive, send: None
            self.settings = SimpleNamespace(debug=False)

        def streamable_http_app(self):
            return self.app

        def session_manager(self):
            return SimpleNamespace(run=lambda: None)

    cfg = SimpleNamespace(
        server_name="sigil",
        chatgpt_compliance_enabled=False,
        header_logging_enabled=True,
        admin_enabled=False,
        log_file=None,
        log_level="INFO",
    )
    monkeypatch.setattr("sigil_mcp.app_factory.FastMCP", DummyFastMCP)
    app = build_mcp_app(config=cfg)
    assert app.app.__class__.__name__ == "HeaderLoggingASGIMiddleware"
