# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

from types import SimpleNamespace

from sigil_mcp.app_factory import (
    ChatGPTComplianceMiddleware,
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
