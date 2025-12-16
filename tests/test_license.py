import os

from sigil_mcp import license as license_mod


def test_is_pro_active_env(monkeypatch):
    monkeypatch.delenv("SIGIL_MCP_PRO_ACTIVE", raising=False)
    assert license_mod.is_pro_active() is False

    monkeypatch.setenv("SIGIL_MCP_PRO_ACTIVE", "1")
    assert license_mod.is_pro_active() is True


def test_require_pro_feature_decorator(monkeypatch):
    called = {}

    @license_mod.require_pro_feature("test-feature")
    def fn(x):
        called['v'] = x
        return x * 2

    # Not active -> raises
    monkeypatch.delenv("SIGIL_MCP_PRO_ACTIVE", raising=False)
    try:
        raised = False
        fn(2)
    except PermissionError:
        raised = True
    assert raised

    # Active -> works
    monkeypatch.setenv("SIGIL_MCP_PRO_ACTIVE", "true")
    assert fn(3) == 6
    assert called['v'] == 3
