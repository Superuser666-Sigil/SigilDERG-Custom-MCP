# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Authentication middleware for Sigil MCP Server.

Provides API key authentication to secure the MCP server when exposed via ngrok.
"""

import hashlib
import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

_api_key_path: Path | None = None
_WORKSPACE_FALLBACK = Path.cwd() / ".sigil_mcp_server" / "api_key"


def get_api_key_path() -> Path:
    """
    Resolve the API key file path with environment overrides and sandbox fallback.

    Priority:
    1) SIGIL_MCP_API_KEY_FILE (absolute or relative path)
    2) SIGIL_MCP_HOME/<api_key> (defaults to ~/.sigil_mcp_server/api_key)
    3) Workspace-local fallback under ./.sigil_mcp_server/api_key when writes are blocked.
    """
    global _api_key_path

    if _api_key_path is not None:
        return _api_key_path

    env_path = os.getenv("SIGIL_MCP_API_KEY_FILE")
    if env_path:
        _api_key_path = Path(env_path).expanduser().resolve()
        return _api_key_path

    base_dir = Path(
        os.getenv("SIGIL_MCP_HOME", Path.home() / ".sigil_mcp_server")
    ).expanduser()
    _api_key_path = (base_dir / "api_key").resolve()
    return _api_key_path


# API key file location (exposed for compatibility; functions call get_api_key_path())
API_KEY_FILE = get_api_key_path()


def _update_api_key_path(new_path: Path) -> None:
    """Update the cached API key path and exported constant."""
    global _api_key_path, API_KEY_FILE
    _api_key_path = new_path
    API_KEY_FILE = new_path


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def initialize_api_key() -> str | None:
    """
    Initialize API key authentication.

    Returns the API key (only time it's displayed in plaintext).
    Returns None if key already exists.
    """
    path = get_api_key_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.info(f"API key file already exists at {path}")
        logger.warning("If you've lost your API key, delete this file and restart the server")
        with open(path) as f:
            return None  # Don't return existing key

    # Generate new API key
    api_key = generate_api_key()
    api_key_hash = hash_api_key(api_key)

    # Store hash with fallback for sandboxed environments
    try:
        with open(path, 'w') as f:
            f.write(api_key_hash)
        path.chmod(0o600)
        logger.info(f"Generated new API key and stored hash at {path}")
        return api_key
    except PermissionError:
        fallback = _WORKSPACE_FALLBACK.resolve()
        fallback.parent.mkdir(parents=True, exist_ok=True)
        with open(fallback, 'w') as f:
            f.write(api_key_hash)
        fallback.chmod(0o600)
        _update_api_key_path(fallback)
        logger.warning(
            "Permission denied writing API key to %s; using workspace-local path %s instead",
            path,
            fallback,
        )
        return api_key


def verify_api_key(provided_key: str) -> bool:
    """
    Verify an API key against the stored hash.

    Args:
        provided_key: The API key to verify

    Returns:
        True if the key is valid, False otherwise
    """
    path = get_api_key_path()

    if not path.exists():
        logger.warning("No API key file found - authentication disabled")
        return True  # Allow access if auth not configured

    try:
        with open(path) as f:
            stored_hash = f.read().strip()

        provided_hash = hash_api_key(provided_key)
        return secrets.compare_digest(stored_hash, provided_hash)

    except PermissionError:
        logger.error("Permission denied reading API key file at %s", path)
        return False
    except Exception as e:
        logger.error(f"Error verifying API key: {e}")
        return False


def get_api_key_from_env() -> str | None:
    """Get API key from environment variable."""
    return os.environ.get("SIGIL_MCP_API_KEY")
