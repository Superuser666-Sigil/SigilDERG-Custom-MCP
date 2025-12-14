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
        self._config_path: Optional[Path] = None
        self._load_config(config_path)
        self._mode = self._resolve_mode()
        self._apply_mode_defaults()
        self._validate_embeddings_dimension()
        self._warn_if_insecure_prod()
    
    def _resolve_mode(self) -> str:
        """Resolve deployment mode from env or config with a safe default."""

        raw_mode = os.getenv("SIGIL_MCP_MODE") or self.config_data.get("mode") or "dev"
        mode = str(raw_mode).strip().lower()
        if mode not in {"dev", "prod"}:
            logger.warning(
                "Invalid SIGIL_MCP_MODE '%s'; defaulting to 'dev'", raw_mode
            )
            mode = "dev"
        # Persist the normalized mode back to config_data for downstream consumers
        self.config_data["mode"] = mode
        return mode

    def _apply_mode_defaults(self) -> None:
        """Apply security-sensitive defaults based on deployment mode."""

        auth_cfg = self.config_data.setdefault("authentication", {})
        admin_cfg = self.config_data.setdefault("admin", {})
        mcp_cfg = self.config_data.setdefault("mcp_server", {})
        admin_ui_cfg = self.config_data.setdefault("admin_ui", {})
        embeddings_cfg = self.config_data.setdefault("embeddings", {})
        self.config_data.setdefault("external_mcp_servers", [])
        self.config_data.setdefault("external_mcp_auto_install", False)

        dev_defaults = {
            "allow_local_bypass": True,
            "enabled": False,
            "oauth_enabled": True,
            "require_api_key": False,
            "admin_ui_autostart": True,
        }
        prod_defaults = {
            "allow_local_bypass": False,
            "enabled": True,
            "oauth_enabled": True,
            "require_api_key": True,
            "admin_ui_autostart": False,
        }
        defaults = dev_defaults if self._mode == "dev" else prod_defaults

        auth_cfg.setdefault("allow_local_bypass", defaults["allow_local_bypass"])
        auth_cfg.setdefault("enabled", defaults["enabled"])
        auth_cfg.setdefault("oauth_enabled", defaults["oauth_enabled"])
        auth_cfg.setdefault("allowed_ips", auth_cfg.get("allowed_ips", []))

        admin_cfg.setdefault("require_api_key", defaults["require_api_key"])
        admin_cfg.setdefault("allowed_ips", admin_cfg.get("allowed_ips", ["127.0.0.1", "::1"]))

        # MCP transport defaults
        mcp_cfg.setdefault("sse_path", "/mcp/sse")
        mcp_cfg.setdefault("http_path", "/")
        mcp_cfg.setdefault("message_path", "/mcp/messages/")
        mcp_cfg.setdefault("require_token", False)
        mcp_cfg.setdefault("token", mcp_cfg.get("token"))

        admin_ui_cfg.setdefault("auto_start", defaults["admin_ui_autostart"])
        admin_ui_cfg.setdefault("path", "./sigil-admin-ui")
        admin_ui_cfg.setdefault("command", "npm")
        admin_ui_cfg.setdefault("args", ["run", "dev"])
        admin_ui_cfg.setdefault("port", 5173)
        embeddings_cfg.setdefault("enabled", False)
        # Optional llama.cpp tuning parameters for local embedding provider
        embeddings_cfg.setdefault("llamacpp_threads", None)
        embeddings_cfg.setdefault("llamacpp_batch_size", None)
        embeddings_cfg.setdefault("llamacpp_n_batch", None)
        embeddings_cfg.setdefault("llamacpp_n_ubatch", None)
        embeddings_cfg.setdefault("llamacpp_threads_batch", None)

    def _warn_if_insecure_prod(self) -> None:
        """Log warnings when production mode uses insecure overrides."""

        if self._mode != "prod":
            return

        if self.allow_local_bypass:
            logger.warning(
                "Production mode with authentication.allow_local_bypass enabled - "
                "local requests will skip authentication."
            )

        if not self.auth_enabled:
            logger.warning(
                "Production mode with authentication disabled - all requests will bypass auth."
            )

        if not self.admin_require_api_key:
            logger.warning(
                "Production mode with admin.require_api_key disabled - admin endpoints allow unauthenticated access."
            )

        if not self.admin_api_key:
            logger.warning(
                "Production mode without admin.api_key configured - admin API will be unavailable or insecure."
            )

        if not self.allowed_ips:
            logger.warning(
                "Production mode without authentication.allowed_ips configured - "
                "all client IPs are permitted."
            )

    def _load_config(self, config_path: Optional[Path] = None):
        """Load configuration from file or environment."""
        # Try specified path first
        if config_path:
            if config_path.exists():
                self._load_from_file(config_path)
                self._config_path = config_path
                return
            else:
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
            self._config_path = local_config
            return
        
        # Try user config directory
        user_config = Path.home() / ".sigil_mcp_server" / "config.json"
        if user_config.exists():
            self._load_from_file(user_config)
            self._config_path = user_config
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
            self._config_path = path
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")
            self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables (backward compatibility)."""
        self.config_data = {
            "mode": os.getenv("SIGIL_MCP_MODE", "dev"),
            "server": {
                "name": os.getenv("SIGIL_MCP_NAME", "sigil_repos"),
                "host": os.getenv("SIGIL_MCP_HOST", "127.0.0.1"),
                "port": int(os.getenv("SIGIL_MCP_PORT", "8000")),
                "log_level": os.getenv("SIGIL_MCP_LOG_LEVEL", "INFO"),
                "chatgpt_compliance_enabled": (
                    os.getenv("SIGIL_MCP_CHATGPT_COMPLIANCE_ENABLED", "true").lower()
                    == "true"
                ),
                "header_logging_enabled": (
                    os.getenv("SIGIL_MCP_HEADER_LOGGING_ENABLED", "true").lower()
                    == "true"
                ),
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
                    # Common VCS / editor / virtualenv / cache dirs
                    ".git", "__pycache__", "node_modules", "target",
                    "build", "dist", ".venv", "venv", ".tox",
                    ".mypy_cache", ".pytest_cache", "coverage", ".coverage",
                    "htmlcov", "env", ".env", "out", "bin", "obj",
                    "pkg", "vendor", "deps", ".gradle", ".idea", ".vscode",
                    # Language/tool specific build dirs
                    "cmake-build-debug", "cmake-build-release", "dist-newstyle",
                    "_build", "deps/_build"
                ],
                "ignore_extensions": [
                    # Python bytecode / extension modules
                    ".pyc", ".pyo", ".pyd",
                    # Native objects / archives
                    ".so", ".o", ".a", ".dll", ".dylib", ".rlib", ".rmeta",
                    # Executables / binaries
                    ".exe", ".bin",
                    # Java / JVM artifacts
                    ".class", ".jar", ".war", ".ear",
                    # Web / image / fonts / assets
                    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg",
                    ".ico", ".woff", ".woff2", ".ttf",
                    # Archives / compressed
                    ".zip", ".tar", ".gz", ".bz2", ".xz",
                    # Other language/tool artifacts
                    ".hi", ".beam", ".dll", ".so", ".rlib"
                ],
            },
            "repositories": self._parse_repo_map(os.getenv("SIGIL_REPO_MAP", "")),
            "index": {
                "path": os.getenv("SIGIL_INDEX_PATH", "~/.sigil_index")
            },
            "external_mcp_auto_install": os.getenv("SIGIL_MCP_AUTO_INSTALL", "false").lower() == "true",
            "external_mcp_servers": self._load_external_mcp_servers_from_env(),
            "mcp_server": self._load_mcp_server_from_env(),
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

    def _load_mcp_server_from_env(self) -> Dict[str, Any]:
        """Load MCP transport settings from environment variables."""
        return {
            "sse_path": os.getenv("SIGIL_MCP_SSE_PATH", "/mcp/sse"),
            "http_path": os.getenv("SIGIL_MCP_HTTP_PATH", "/"),
            "message_path": os.getenv("SIGIL_MCP_MESSAGE_PATH", "/mcp/messages/"),
            "require_token": os.getenv("SIGIL_MCP_REQUIRE_TOKEN", "false").lower() == "true",
            "token": os.getenv("SIGIL_MCP_SERVER_TOKEN"),
        }

    def _load_admin_ui_from_env(self) -> Dict[str, Any]:
        """Load admin UI autostart settings from environment variables."""
        args_raw = os.getenv("SIGIL_ADMIN_UI_ARGS", "")
        args = [a for a in args_raw.split() if a] if args_raw else []
        return {
            "auto_start": os.getenv("SIGIL_ADMIN_UI_AUTOSTART", "true").lower() == "true",
            "path": os.getenv("SIGIL_ADMIN_UI_PATH", "./sigil-admin-ui"),
            "command": os.getenv("SIGIL_ADMIN_UI_COMMAND", "npm"),
            "args": args or ["run", "dev"],
            "port": int(os.getenv("SIGIL_ADMIN_UI_PORT", "5173")),
        }

    def _load_external_mcp_servers_from_env(self) -> list[dict]:
        """
        Load external MCP server definitions from SIGIL_MCP_SERVERS (JSON array).
        """
        raw = os.getenv("SIGIL_MCP_SERVERS")
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            logger.warning("SIGIL_MCP_SERVERS must be a JSON array; got %s", type(parsed))
        except Exception as exc:
            logger.warning("Failed to parse SIGIL_MCP_SERVERS: %s", exc)
        return []
    
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
    def chatgpt_compliance_enabled(self) -> bool:
        return bool(self.get("server.chatgpt_compliance_enabled", True))

    @property
    def header_logging_enabled(self) -> bool:
        return bool(self.get("server.header_logging_enabled", True))
    
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
    def mcp_sse_path(self) -> str:
        """Path for SSE MCP transport."""
        return str(self.get("mcp_server.sse_path", "/mcp/sse"))

    @property
    def mcp_http_path(self) -> str:
        """Path for streamable HTTP MCP transport."""
        return str(self.get("mcp_server.http_path", "/"))

    @property
    def mcp_message_path(self) -> str:
        """Path prefix for MCP message routing."""
        return str(self.get("mcp_server.message_path", "/mcp/messages/"))

    @property
    def mcp_require_token(self) -> bool:
        """Whether to require a bearer token for MCP transports."""
        return bool(self.get("mcp_server.require_token", False))

    @property
    def mcp_server_token(self) -> Optional[str]:
        """Bearer token required for MCP transports when enabled."""
        token = self.get("mcp_server.token")
        if not token:
            token = os.getenv("SIGIL_MCP_SERVER_TOKEN")
        return token

    @property
    def admin_ui_auto_start(self) -> bool:
        return bool(self.get("admin_ui.auto_start", True))

    @property
    def admin_ui_path(self) -> str:
        return str(self.get("admin_ui.path", "./sigil-admin-ui"))

    @property
    def admin_ui_command(self) -> str:
        return str(self.get("admin_ui.command", "npm"))

    @property
    def admin_ui_args(self) -> list[str]:
        return list(self.get("admin_ui.args", ["run", "dev"]))

    @property
    def admin_ui_port(self) -> int:
        try:
            return int(self.get("admin_ui.port", 5173))
        except (TypeError, ValueError):
            return 5173

    @property
    def external_mcp_auto_install(self) -> bool:
        return bool(self.get("external_mcp_auto_install", False))

    @property
    def external_mcp_servers(self) -> list[dict]:
        """External MCP servers configuration list."""
        return self.get("external_mcp_servers", [])

    @property
    def mode(self) -> str:
        return self._mode
    
    @property
    def auth_enabled(self) -> bool:
        default = False if self._mode == "dev" else True
        return bool(self.get("authentication.enabled", default))
    
    @property
    def oauth_enabled(self) -> bool:
        default = True
        return bool(self.get("authentication.oauth_enabled", default))
    
    @property
    def allow_local_bypass(self) -> bool:
        default = True if self._mode == "dev" else False
        return bool(self.get("authentication.allow_local_bypass", default))
    
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
        return self.get("embeddings.provider", "llamacpp")

    @property
    def embeddings_model(self) -> Optional[str]:
        """Get embedding model name or path."""
        return self.get(
            "embeddings.model",
            "/home/dave/models/jina/jina-embeddings-v2-base-code-Q4_K_M.gguf",
        )

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
        kwargs = {k: v for k, v in embeddings_config.items() if k not in known_keys and v is not None}
        # Map convenience keys to llama.cpp arguments and drop the old names
        if "llamacpp_threads" in kwargs:
            kwargs.setdefault("n_threads", kwargs.pop("llamacpp_threads"))
        if "llamacpp_batch_size" in kwargs:
            kwargs.setdefault("batch_size", kwargs.pop("llamacpp_batch_size"))
        if "llamacpp_threads_batch" in kwargs:
            kwargs.setdefault("n_threads_batch", kwargs.pop("llamacpp_threads_batch"))
        if "llamacpp_n_batch" in kwargs:
            kwargs.setdefault("n_batch", kwargs.pop("llamacpp_n_batch"))
        if "llamacpp_n_ubatch" in kwargs:
            kwargs.setdefault("n_ubatch", kwargs.pop("llamacpp_n_ubatch"))
        if "llamacpp_context_size" in kwargs:
            kwargs.setdefault("context_size", kwargs.pop("llamacpp_context_size"))
        if "n_ctx" in kwargs:
            kwargs.setdefault("context_size", kwargs.pop("n_ctx"))
        # Strip any remaining None values before passing through
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # Provide a sane default for llama.cpp GPU offload if not explicitly set
        kwargs.setdefault("n_gpu_layers", self.embeddings_n_gpu_layers)
        return kwargs

    @property
    def embeddings_bucket_thresholds(self) -> list[int]:
        """Get bucketing thresholds (token counts) for embedding batching.

        Returns list of ascending integer thresholds used to partition texts
        into buckets (e.g. [256,512,1024,2048]).
        """
        return list(self.get("embeddings.bucket_thresholds", [512, 1024, 2048]))

    @property
    def embeddings_target_tokens(self) -> int:
        """Preferred tokens per chunk when tokenization is available."""
        try:
            return int(self.get("embeddings.target_tokens", 1024))
        except (TypeError, ValueError):
            return 1024

    @property
    def embeddings_max_tokens(self) -> int:
        """Hard maximum tokens per chunk; trigger token-based hard-wrap when exceeded."""
        try:
            return int(self.get("embeddings.max_tokens", 2048))
        except (TypeError, ValueError):
            return 2048

    @property
    def embeddings_token_overlap(self) -> int:
        """Token overlap to use when performing token-based hard wrapping."""
        try:
            return int(self.get("embeddings.token_overlap", 128))
        except (TypeError, ValueError):
            return 128

    @property
    def embed_hard_chars(self) -> int:
        """Maximum characters allowed in a single chunk before applying hard wrap."""
        try:
            return int(self.get("embeddings.hard_chars", 12000))
        except (TypeError, ValueError):
            return 12000

    @property
    def embed_hard_window(self) -> int:
        """Window size (chars) to use when hard-wrapping oversized chunks."""
        try:
            return int(self.get("embeddings.hard_window", 10000))
        except (TypeError, ValueError):
            return 10000

    @property
    def embed_hard_overlap(self) -> int:
        """Overlap (chars) between subsequent hard-wrapped windows."""
        try:
            return int(self.get("embeddings.hard_overlap", 1000))
        except (TypeError, ValueError):
            return 1000

    @property
    def embeddings_include_solution(self) -> bool:
        """Whether to include solution/answer text when building canonical JSONL embeddings."""
        return bool(self.get("embeddings.include_solution", True))

    @property
    def embeddings_n_gpu_layers(self) -> int:
        """Default GPU offload layers for llama.cpp embeddings."""
        raw = self.get("embeddings.n_gpu_layers", 999)
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid embeddings.n_gpu_layers '%s', defaulting to 999", raw
            )
            return 999
    
    @property
    def repositories(self) -> Dict[str, str]:
        return self.get("repositories", {})

    @property
    def repositories_config(self) -> Dict[str, dict]:
        """Return repositories configured with optional per-repo options.

        Accepts two forms in config.json:
        1) "repositories": { "name": "/abs/path" }
        2) "repositories": { "name": { "path": "/abs/path", "respect_gitignore": true } }

        The returned mapping is: name -> {"path": str, "respect_gitignore": bool}
        """
        raw = self.get("repositories", {}) or {}
        parsed: Dict[str, dict] = {}
        for name, value in raw.items():
            try:
                if isinstance(value, str):
                    parsed[name] = {"path": value, "respect_gitignore": True, "ignore_patterns": []}
                elif isinstance(value, dict):
                    path = value.get("path") or value.get("repo_path")
                    if not path:
                        continue
                    parsed[name] = {
                        "path": path,
                        "respect_gitignore": bool(value.get("respect_gitignore", True)),
                        "ignore_patterns": list(value.get("ignore_patterns", []) or []),
                    }
                else:
                    # unknown format; skip
                    continue
            except Exception:
                continue
        return parsed
    
    @property
    def index_path(self) -> Path:
        path_str = self.get("index.path", "~/.sigil_index")
        return Path(path_str).expanduser().resolve()

    @property
    def index_ignore_patterns(self) -> list[str]:
        """Get global ignore patterns used by the indexer.

        This returns the raw list from `index.ignore_patterns` in `config.json`.
        Patterns may include simple glob expressions (e.g. "*.pyc"),
        directory suffixes (e.g. ".git/"), and negations beginning with "!"
        (e.g. "!/sigil-admin-ui/src/lib/") to re-include specific paths.
        """
        return list(self.get("index.ignore_patterns", []))

    @property
    def embeddings_llamacpp_threads(self) -> int | None:
        """Optional number of threads to pass to llama.cpp provider (None = auto)."""
        val = self.get("embeddings.llamacpp_threads", None)
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            logger.warning("Invalid embeddings.llamacpp_threads '%s' in config", val)
            return None

    @property
    def embeddings_llamacpp_batch_size(self) -> int | None:
        """Optional batch size to use when calling llama.cpp embed APIs."""
        val = self.get("embeddings.llamacpp_batch_size", None)
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            logger.warning("Invalid embeddings.llamacpp_batch_size '%s' in config", val)
            return None

    @property
    def embeddings_llamacpp_threads_batch(self) -> int | None:
        """Optional n_threads_batch to use for llama.cpp embeddings (None = library default)."""
        val = self.get("embeddings.llamacpp_threads_batch", None)
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            logger.warning("Invalid embeddings.llamacpp_threads_batch '%s' in config", val)
            return None

    @property
    def embeddings_llamacpp_n_batch(self) -> int | None:
        """Optional n_batch (tokens per eval) for llama.cpp embeddings."""
        val = self.get("embeddings.llamacpp_n_batch", None)
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            logger.warning("Invalid embeddings.llamacpp_n_batch '%s' in config", val)
            return None

    @property
    def embeddings_llamacpp_n_ubatch(self) -> int | None:
        """Optional n_ubatch (micro-batch) for llama.cpp embeddings."""
        val = self.get("embeddings.llamacpp_n_ubatch", None)
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            logger.warning("Invalid embeddings.llamacpp_n_ubatch '%s' in config", val)
            return None

    @property
    def lance_dir(self) -> Path:
        """Directory used by LanceDB, defaulting to ``index_dir / "lancedb"``."""
        path_str = self.get("index.lance_dir")
        if path_str:
            return Path(path_str).expanduser().resolve()
        return self.index_path / "lancedb"

    @property
    def index_allow_vector_schema_overwrite(self) -> bool:
        """Allow overwriting the vector table on embedding dimension mismatch."""
        return bool(self.get("index.allow_vector_schema_overwrite", True))
    
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
        default = False if self._mode == "dev" else True
        return bool(self.get("admin.require_api_key", default))


# Global config instance
_config: Optional[Config] = None
_config_env_signature: Optional[tuple[str, ...]] = None
_ENV_SENSITIVE_KEYS = [
    "SIGIL_MCP_MODE",
    "SIGIL_INDEX_PATH",
    "SIGIL_MCP_WATCH_ENABLED",
    "SIGIL_MCP_LOG_FILE",
    "SIGIL_MCP_ADMIN_API_KEY",
    "SIGIL_MCP_ADMIN_REQUIRE_API_KEY",
    "SIGIL_MCP_ADMIN_ALLOWED_IPS",
    "SIGIL_MCP_EMBEDDINGS_ENABLED",
    "SIGIL_MCP_EMBEDDINGS_PROVIDER",
    "SIGIL_MCP_EMBEDDINGS_MODEL",
]


def _capture_env_signature() -> tuple[str, ...]:
    """Capture relevant environment values to detect changes in tests."""
    return tuple(os.getenv(key, "") or "" for key in _ENV_SENSITIVE_KEYS)


def get_config() -> Config:
    """Get or create global configuration instance."""
    global _config, _config_env_signature
    if _config is None:
        _config = Config()
        _config_env_signature = _capture_env_signature()
        return _config

    if os.getenv("PYTEST_CURRENT_TEST"):
        current_sig = _capture_env_signature()
        if current_sig != _config_env_signature:
            _config = Config()
            _config_env_signature = current_sig

    return _config


def load_config(config_path: Optional[Path] = None):
    """Load configuration from specified path."""
    global _config, _config_env_signature
    _config = Config(config_path)
    _config_env_signature = _capture_env_signature()
    return _config


def save_config(cfg: Config, target_path: Optional[Path] = None) -> Path:
    """Persist the given Config object's data to a JSON file.

    By default, this will write to ./config.json if it exists, otherwise
    to ./config.json in the current working directory (creating it if needed).
    Returns the Path written to.
    """
    # Determine target path
    if target_path:
        out_path = target_path
    else:
        # Prefer the original config path the Config was loaded from, otherwise cwd/config.json
        out_path = getattr(cfg, "_config_path", None) or (Path.cwd() / "config.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def _deep_merge(existing: dict, new: dict) -> dict:
        """Recursively merge new into existing and return existing."""
        for k, v in new.items():
            if k in existing and isinstance(existing[k], dict) and isinstance(v, dict):
                _deep_merge(existing[k], v)
            else:
                existing[k] = v
        return existing

    try:
        # If file exists, load and merge rather than overwrite
        if out_path.exists():
            try:
                with out_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
            except Exception:
                existing = {}

            # Create a timestamped backup before modifying and rotate old backups
            try:
                from datetime import datetime

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = out_path.with_name(f"{out_path.name}.{ts}.bak")
                with out_path.open("r", encoding="utf-8") as fsrc, backup_path.open("w", encoding="utf-8") as fdst:
                    fdst.write(fsrc.read())
                logger.info("Backed up existing config to %s", backup_path)

                # Rotate backups: keep only the most recent N backups
                try:
                    max_backups = 5
                    backups = sorted([p for p in out_path.parent.glob(f"{out_path.name}.*.bak") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
                    for old in backups[max_backups:]:
                        try:
                            old.unlink()
                            logger.info("Removed old backup %s", old)
                        except Exception:
                            logger.debug("Failed to remove old backup %s", old, exc_info=True)
                except Exception:
                    logger.debug("Backup rotation failed", exc_info=True)
            except Exception:
                logger.exception("Failed to create backup of existing config %s", out_path)

            merged = _deep_merge(existing, cfg.config_data or {})
            to_write = merged
        else:
            to_write = cfg.config_data or {}

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(to_write, f, indent=2)
        logger.info("Saved configuration to %s", out_path)
    except Exception as exc:
        logger.exception("Failed to save configuration to %s: %s", out_path, exc)
        raise

    return out_path
