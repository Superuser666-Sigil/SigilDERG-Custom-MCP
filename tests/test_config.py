# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Tests for configuration module (config.py).
"""

import json
import os
from pathlib import Path
from sigil_mcp.config import Config, get_config, load_config


class TestConfigLoading:
    """Test configuration loading from various sources."""
    
    def test_load_config_from_file(self, test_config_file):
        """Test loading configuration from JSON file."""
        config = Config(test_config_file)
        assert config.server_name == "test_server"
        assert config.server_host == "127.0.0.1"
        assert config.server_port == 8000
        assert config.log_level == "DEBUG"
    
    def test_load_config_from_current_directory(self, temp_dir, monkeypatch):
        """Test loading config.json from current directory."""
        monkeypatch.chdir(temp_dir)
        
        config_path = temp_dir / "config.json"
        config_data = {
            "server": {"name": "local_server", "port": 9000}
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        config = Config()
        assert config.server_name == "local_server"
        assert config.server_port == 9000
    
    def test_load_config_fallback_to_env(self, monkeypatch):
        """Test fallback to environment variables when no config file."""
        monkeypatch.setenv("SIGIL_MCP_NAME", "env_server")
        monkeypatch.setenv("SIGIL_MCP_PORT", "7000")
        monkeypatch.setenv("SIGIL_MCP_LOG_LEVEL", "ERROR")
        
        # Use non-existent path to force env fallback
        config = Config(Path("/nonexistent/config.json"))
        assert config.server_name == "env_server"
        assert config.server_port == 7000
        assert config.log_level == "ERROR"
    
    def test_load_config_with_repo_map(self, monkeypatch):
        """Test parsing SIGIL_REPO_MAP environment variable."""
        monkeypatch.setenv("SIGIL_REPO_MAP", "repo1:/path/to/repo1;repo2:/path/to/repo2")
        
        config = Config(Path("/nonexistent/config.json"))
        repos = config.repositories
        assert "repo1" in repos
        assert repos["repo1"] == "/path/to/repo1"
        assert "repo2" in repos
        assert repos["repo2"] == "/path/to/repo2"


class TestConfigProperties:
    """Test configuration property accessors."""
    
    def test_server_properties(self, test_config_file):
        """Test server-related properties."""
        config = Config(test_config_file)
        assert config.server_name == "test_server"
        assert config.server_host == "127.0.0.1"
        assert config.server_port == 8000
        assert config.log_level == "DEBUG"
    
    def test_auth_properties(self, test_config_file):
        """Test authentication-related properties."""
        config = Config(test_config_file)
        assert config.auth_enabled is True
        assert config.oauth_enabled is True
        assert config.allow_local_bypass is True
        assert config.allowed_ips == ["192.168.1.1"]
    
    def test_repositories_property(self, test_config_file):
        """Test repositories property."""
        config = Config(test_config_file)
        repos = config.repositories
        assert isinstance(repos, dict)
        assert "test_repo" in repos
    
    def test_index_path_property(self, test_config_file):
        """Test index path property."""
        config = Config(test_config_file)
        index_path = config.index_path
        assert isinstance(index_path, Path)
        assert index_path.is_absolute()
    
    def test_index_path_expansion(self, temp_dir):
        """Test that ~ is expanded in index path."""
        config_path = temp_dir / "config.json"
        config_data = {"index": {"path": "~/.test_index"}}
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        config = Config(config_path)
        assert "~" not in str(config.index_path)
        assert config.index_path.is_absolute()


class TestConfigGet:
    """Test configuration get method."""
    
    def test_get_nested_value(self, test_config_file):
        """Test getting nested configuration values."""
        config = Config(test_config_file)
        assert config.get("server.name") == "test_server"
        assert config.get("server.port") == 8000
        assert config.get("authentication.enabled") is True
    
    def test_get_with_default(self, test_config_file):
        """Test get method with default value."""
        config = Config(test_config_file)
        assert config.get("nonexistent.key", "default") == "default"
        assert config.get("server.nonexistent", 999) == 999
    
    def test_get_deeply_nested(self, temp_dir):
        """Test getting deeply nested values."""
        config_path = temp_dir / "config.json"
        config_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep"
                    }
                }
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        config = Config(config_path)
        assert config.get("level1.level2.level3.value") == "deep"
    
    def test_get_returns_none_for_missing(self, test_config_file):
        """Test that get returns None for missing keys without default."""
        config = Config(test_config_file)
        assert config.get("missing.key") is None


class TestExternalMCPConfig:
    def test_external_mcp_default_empty(self, monkeypatch):
        for key in list(os.environ.keys()):
            if key.startswith("SIGIL_MCP_SERVERS"):
                monkeypatch.delenv(key, raising=False)
        config = Config(Path("/nonexistent/config.json"))
        assert config.external_mcp_servers == []

    def test_external_mcp_env_parse(self, monkeypatch):
        servers = [
            {"name": "playwright", "type": "streamable-http", "url": "http://localhost:3001/"},
            {"name": "local", "type": "stdio", "command": "python", "args": ["tool.py"]},
        ]
        monkeypatch.setenv("SIGIL_MCP_SERVERS", json.dumps(servers))
        config = Config(Path("/nonexistent/config.json"))
        assert config.external_mcp_servers == servers


class TestConfigDefaults:
    """Test default configuration values."""
    
    def test_default_server_values(self, monkeypatch):
        """Test default server configuration values."""
        # Clear all env vars
        for key in list(os.environ.keys()):
            if key.startswith("SIGIL_"):
                monkeypatch.delenv(key, raising=False)
        
        config = Config(Path("/nonexistent/config.json"))
        assert config.server_name == "sigil_repos"
        assert config.server_host == "127.0.0.1"
        assert config.server_port == 8000
        assert config.log_level == "INFO"
    
    def test_default_auth_values(self, monkeypatch):
        """Test default authentication configuration values."""
        for key in list(os.environ.keys()):
            if key.startswith("SIGIL_"):
                monkeypatch.delenv(key, raising=False)
        
        config = Config(Path("/nonexistent/config.json"))
        assert config.auth_enabled is True
        assert config.oauth_enabled is True
        assert config.allow_local_bypass is True
        assert config.allowed_ips == []


class TestGlobalConfigInstance:
    """Test global configuration instance management."""
    
    def test_get_config_singleton(self):
        """Test that get_config returns singleton instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_load_config_updates_global(self, test_config_file):
        """Test that load_config updates global instance."""
        config = load_config(test_config_file)
        assert config.server_name == "test_server"
        
        # Get global instance
        global_config = get_config()
        assert global_config.server_name == "test_server"
        assert global_config is config
