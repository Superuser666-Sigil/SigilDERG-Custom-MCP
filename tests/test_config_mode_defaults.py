import json

from sigil_mcp.config import Config


def test_dev_mode_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("SIGIL_MCP_MODE", raising=False)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({}))

    cfg = Config(cfg_path)

    assert cfg.mode == "dev"
    assert cfg.allow_local_bypass is True
    assert cfg.auth_enabled is False
    assert cfg.admin_require_api_key is False


def test_prod_mode_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGIL_MCP_MODE", "prod")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({}))

    cfg = Config(cfg_path)

    assert cfg.mode == "prod"
    assert cfg.allow_local_bypass is False
    assert cfg.auth_enabled is True
    assert cfg.admin_require_api_key is True
