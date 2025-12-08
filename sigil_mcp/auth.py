# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Authentication middleware for Sigil MCP Server.

Provides API key authentication to secure the MCP server when exposed via ngrok.
"""

import os
import secrets
import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# API key file location
API_KEY_FILE = Path.home() / ".sigil_mcp_server" / "api_key"


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def initialize_api_key() -> Optional[str]:
    """
    Initialize API key authentication.
    
    Returns the API key (only time it's displayed in plaintext).
    Returns None if key already exists.
    """
    API_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    if API_KEY_FILE.exists():
        logger.info(f"API key file already exists at {API_KEY_FILE}")
        logger.warning("If you've lost your API key, delete this file and restart the server")
        with open(API_KEY_FILE, 'r') as f:
            return None  # Don't return existing key
    
    # Generate new API key
    api_key = generate_api_key()
    api_key_hash = hash_api_key(api_key)
    
    # Store hash
    with open(API_KEY_FILE, 'w') as f:
        f.write(api_key_hash)
    
    # Set restrictive permissions
    API_KEY_FILE.chmod(0o600)
    
    logger.info(f"Generated new API key and stored hash at {API_KEY_FILE}")
    return api_key


def verify_api_key(provided_key: str) -> bool:
    """
    Verify an API key against the stored hash.
    
    Args:
        provided_key: The API key to verify
    
    Returns:
        True if the key is valid, False otherwise
    """
    if not API_KEY_FILE.exists():
        logger.warning("No API key file found - authentication disabled")
        return True  # Allow access if auth not configured
    
    try:
        with open(API_KEY_FILE, 'r') as f:
            stored_hash = f.read().strip()
        
        provided_hash = hash_api_key(provided_key)
        return secrets.compare_digest(stored_hash, provided_hash)
    
    except Exception as e:
        logger.error(f"Error verifying API key: {e}")
        return False


def get_api_key_from_env() -> Optional[str]:
    """Get API key from environment variable."""
    return os.environ.get("SIGIL_MCP_API_KEY")
