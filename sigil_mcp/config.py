# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Configuration loader for Sigil MCP Server.

Loads configuration from config.json file with fallback to environment variables.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def _parse_csv_list(raw_value: Optional[str]) -> list[str]:
    """Parse comma-separated environment variable values into a list."""
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


class Config:
    """Configuration manager for Sigil MCP Server."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.json file. If None, searches in:
                1. ./config.json (current directory)
                2. ~/.sigil_mcp_server/config.json
                3. Falls back to environment variables
        """
        self.config_data: Dict[str, Any] = {}
        self._load_config(config_path)
        self._validate_embeddings_dimension()
    
    def _load_config(self, config_path: Optional[Path] = None):
        """Load configuration from file or environment."""
        # Try specified path first
        if config_path:
            if config_path.exists():
                self._load_from_file(config_path)
                return
            else:
                # Explicit path provided but doesn't exist - skip file-based loading
                # and fall back to environment variables
                logger.info(
                    f"Config path {config_path} does not exist, "
                    "using environment variables"
                )
                self._load_from_env()
                return
        
        # Try current directory
        local_config = Path("config.json")
        if local_config.exists():
            self._load_from_file(local_config)
            return
        
        # Try user config directory
        user_config = Path.home() / ".sigil_mcp_server" / "config.json"
        if user_config.exists():
            self._load_from_file(user_config)
            return
        
        # Fall back to environment variables
        logger.info("No config.json found, using environment variables")
        self._load_from_env()

    def _validate_embeddings_dimension(self) -> None:
        """Validate the configured embeddings dimension and warn on mismatch."""
        dimension_value = self.get("embeddings.dimension")
        if dimension_value is None:
            # Ensure a default is present for downstream consumers
            self.config_data.setdefault("embeddings", {}).setdefault(
                "dimension", 768
            )
            return

        try:
            dimension = int(dimension_value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid embeddings.dimension '%s', defaulting to 768", dimension_value
            )
            self.config_data.setdefault("embeddings", {})["dimension"] = 768
            return

        if dimension != 768:
            logger.warning(
                "Configured embeddings.dimension %s differs from default 768. "
                "Ensure the selected embedding model matches this dimension.",
                dimension,
            )

        self.config_data.setdefault("embeddings", {})["dimension"] = dimension
    
    def _load_from_file(self, path: Path):
        """Load configuration from JSON file."""
        try:
            with open(path, 'r') as f:
                self.config_data = json.load(f)
            logger.info(f"Loaded configuration from {path}")
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")
            self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables (backward compatibility)."""
        self.config_data = {
            "server": {
                "name": os.getenv("SIGIL_MCP_NAME", "sigil_repos"),
                "host": os.getenv("SIGIL_MCP_HOST", "127.0.0.1"),
                "port": int(os.getenv("SIGIL_MCP_PORT", "8000")),
                "log_level": os.getenv("SIGIL_MCP_LOG_LEVEL", "INFO")
            },
            "authentication": {
                "enabled": os.getenv("SIGIL_MCP_AUTH_ENABLED", "true").lower() == "true",
                "oauth_enabled": os.getenv("SIGIL_MCP_OAUTH_ENABLED", "true").lower() == "true",
                "allow_local_bypass": (
                    os.getenv("SIGIL_MCP_ALLOW_LOCAL_BYPASS", "true").lower() == "true"
                ),
                "allowed_ips": _parse_csv_list(os.getenv("SIGIL_MCP_ALLOWED_IPS")),
                "redirect_allow_list": _parse_csv_list(
                    os.getenv("SIGIL_MCP_OAUTH_REDIRECT_ALLOW_LIST")
                ),
            },
            "watch": {
                "enabled": os.getenv("SIGIL_MCP_WATCH_ENABLED", "true").lower() == "true",
                "debounce_seconds": float(os.getenv("SIGIL_MCP_WATCH_DEBOUNCE", "2.0")),
                "ignore_dirs": [
                    ".git", "__pycache__", "node_modules", "target",
                    "build", "dist", ".venv", "venv", ".tox",
                    ".mypy_cache", ".pytest_cache", "coverage", ".coverage"
                ],
                "ignore_extensions": [
                    ".pyc", ".so", ".o", ".a", ".dylib", ".dll",
                    ".exe", ".bin", ".pdf", ".png", ".jpg", ".gif",
                    ".svg", ".ico", ".woff", ".woff2", ".ttf",
                    ".zip", ".tar", ".gz", ".bz2", ".xz"
                ],
            },
            "repositories": self._parse_repo_map(os.getenv("SIGIL_REPO_MAP", "")),
            "index": {
                "path": os.getenv("SIGIL_INDEX_PATH", "~/.sigil_index")
            },
            "admin": self._load_admin_from_env(),
        }

    def _load_admin_from_env(self) -> Dict[str, Any]:
        """Load admin config from environment variables."""
        allowed_ips_raw = os.getenv("SIGIL_MCP_ADMIN_ALLOWED_IPS", "127.0.0.1,::1")
        return {
            "enabled": os.getenv("SIGIL_MCP_ADMIN_ENABLED", "true").lower() == "true",
            "host": os.getenv("SIGIL_MCP_ADMIN_HOST", "127.0.0.1"),
            "port": int(os.getenv("SIGIL_MCP_ADMIN_PORT", "8765")),
            "api_key": os.getenv("SIGIL_MCP_ADMIN_API_KEY") or None,
            "require_api_key": (
                os.getenv("SIGIL_MCP_ADMIN_REQUIRE_API_KEY", "true").lower() == "true"
            ),
            "allowed_ips": _parse_csv_list(allowed_ips_raw),
        }
    
    def _parse_repo_map(self, repo_map_str: str) -> Dict[str, str]:
        """Parse SIGIL_REPO_MAP environment variable format."""
        repos = {}
        if not repo_map_str:
            return repos
        
        for entry in repo_map_str.split(";"):
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            name, path = entry.split(":", 1)
            repos[name.strip()] = path.strip()
        
        return repos
    
    # Getters for easy access
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split(".")
        value = self.config_data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    @property
    def server_name(self) -> str:
        return self.get("server.name", "sigil_repos")
    
    @property
    def server_host(self) -> str:
        return self.get("server.host", "127.0.0.1")
    
    @property
    def server_port(self) -> int:
        return self.get("server.port", 8000)
    
    @property
    def log_level(self) -> str:
        return self.get("server.log_level", "INFO")
    
    @property
    def log_file(self) -> Optional[str]:
        """Get log file path from config or environment."""
        # Check environment variable first
        env_log_file = os.getenv("SIGIL_MCP_LOG_FILE")
        if env_log_file:
            return env_log_file
        # Then check config
        return self.get("server.log_file") or None
    
    @property
    def allowed_hosts(self) -> list[str]:
        """Get allowed Host header values for DNS rebinding protection."""
        return self.get("server.allowed_hosts", ["*"])
    
    @property
    def auth_enabled(self) -> bool:
        return self.get("authentication.enabled", True)
    
    @property
    def oauth_enabled(self) -> bool:
        return self.get("authentication.oauth_enabled", True)
    
    @property
    def allow_local_bypass(self) -> bool:
        return self.get("authentication.allow_local_bypass", True)
    
    @property
    def allowed_ips(self) -> list:
        return self.get("authentication.allowed_ips", [])
    
    @property
    def oauth_redirect_allow_list(self) -> list[str]:
        default_allow_list = [
            "https://chat.openai.com",
            "https://chatgpt.com",
            "https://chat.openai.com/aip/oauth/callback",
            "https://chatgpt.com/aip/oauth/callback",
        ]
        value = self.get("authentication.redirect_allow_list")
        if not value:
            return default_allow_list
        if isinstance(value, str):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]
    
    @property
    def watch_enabled(self) -> bool:
        """Get whether file watching is enabled."""
        return self.get("watch.enabled", True)
    
    @property
    def watch_debounce_seconds(self) -> float:
        """Get file watch debounce time in seconds."""
        return self.get("watch.debounce_seconds", 2.0)
    
    @property
    def watch_ignore_dirs(self) -> list[str]:
        """Get directories to ignore when watching."""
        return self.get("watch.ignore_dirs", [
            # Version control
            ".git",
            # Python
            "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
            "build", "dist", "downloads", "eggs", ".eggs", "lib", "lib64",
            "parts", "sdist", "var", "wheels", ".installed.cfg",
            "develop-eggs", "htmlcov",
            # Virtual environments
            "venv", ".venv", "ENV", "env",
            # IDE
            ".vscode", ".idea",
            # Node.js
            "node_modules",
            # Other build systems
            "target",
            # Sigil runtime
            ".sigil_index", ".sigil_mcp_server",
            # Coverage/testing
            "coverage", ".coverage", ".cache",
        ])
    
    @property
    def watch_ignore_extensions(self) -> list[str]:
        """Get file extensions to ignore when watching."""
        return self.get("watch.ignore_extensions", [
            # Python compiled
            ".pyc", ".pyo", ".pyd",
            # Native/compiled
            ".so", ".o", ".a", ".dylib", ".dll", ".exe", ".bin",
            # Archives
            ".zip", ".tar", ".gz", ".bz2", ".xz", ".egg",
            # JavaScript modules (including Vite temporary files)
            ".mjs",
            # Images
            ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
            # Fonts
            ".woff", ".woff2", ".ttf", ".eot",
            # Documents
            ".pdf",
            # Logs
            ".log",
            # Temp files
            ".tmp", ".temp", ".swp", ".swo",
            # OS files
            ".DS_Store",
        ])
    
    @property
    def embeddings_enabled(self) -> bool:
        """Get whether embeddings are enabled."""
        return self.get("embeddings.enabled", False)
    
    @property
    def embeddings_provider(self) -> Optional[str]:
        """Get embedding provider name."""
        return self.get("embeddings.provider", "jina")

    @property
    def embeddings_model(self) -> Optional[str]:
        """Get embedding model name or path."""
        return self.get("embeddings.model", "jina-embeddings-v2-base")

    @property
    def embeddings_dimension(self) -> int:
        """Get embedding dimension."""
        value = self.get("embeddings.dimension", 768)
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid embeddings.dimension '%s', defaulting to 768", value
            )
            return 768
    
    @property
    def embeddings_cache_dir(self) -> Optional[str]:
        """Get embeddings cache directory."""
        return self.get("embeddings.cache_dir")
    
    @property
    def embeddings_api_key(self) -> Optional[str]:
        """Get embeddings API key (for OpenAI)."""
        api_key = self.get("embeddings.api_key")
        if not api_key:
            # Fall back to OPENAI_API_KEY environment variable
            api_key = os.getenv("OPENAI_API_KEY")
        return api_key
    
    @property
    def embeddings_kwargs(self) -> dict:
        """Get additional embeddings provider kwargs."""
        embeddings_config = self.get("embeddings", {})
        # Return all config except known keys
        known_keys = {
            "enabled",
            "provider",
            "model",
            "dimension",
            "cache_dir",
            "api_key",
        }
        return {k: v for k, v in embeddings_config.items() if k not in known_keys}
    
    @property
    def repositories(self) -> Dict[str, str]:
        return self.get("repositories", {})
    
    @property
    def index_path(self) -> Path:
        path_str = self.get("index.path", "~/.sigil_index")
        return Path(path_str).expanduser().resolve()

    @property
    def lance_dir(self) -> Path:
        """Directory used by LanceDB, defaulting to ``index_dir / "lancedb"``."""
        path_str = self.get("index.lance_dir")
        if path_str:
            return Path(path_str).expanduser().resolve()
        return self.index_path / "lancedb"
    
    # --- Admin API configuration ---

    @property
    def admin_enabled(self) -> bool:
        return self.get("admin.enabled", True)

    @property
    def admin_host(self) -> str:
        return self.get("admin.host", "127.0.0.1")

    @property
    def admin_port(self) -> int:
        return int(self.get("admin.port", 8765))

    @property
    def admin_api_key(self) -> Optional[str]:
        return self.get("admin.api_key")

    @property
    def admin_allowed_ips(self) -> list[str]:
        return self.get("admin.allowed_ips", ["127.0.0.1", "::1"])

    @property
    def admin_require_api_key(self) -> bool:
        return self.get("admin.require_api_key", True)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def load_config(config_path: Optional[Path] = None):
    """Load configuration from specified path."""
    global _config
    _config = Config(config_path)
    return _config
