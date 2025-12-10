from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from ..auth import get_api_key_from_env, verify_api_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AuthSettings:
    """Immutable collection of authentication-related configuration flags."""

    auth_enabled: bool
    oauth_enabled: bool
    allow_local_bypass: bool
    allowed_ips: Sequence[str]


def get_auth_settings(config=None) -> AuthSettings:
    """Create an AuthSettings object from the provided Config (or default)."""

    if config is None:
        from ..config import get_config  # Lazy import to avoid circular deps

        config = get_config()

    allowed_ips = tuple(config.allowed_ips or [])

    return AuthSettings(
        auth_enabled=config.auth_enabled,
        oauth_enabled=config.oauth_enabled,
        allow_local_bypass=config.allow_local_bypass,
        allowed_ips=allowed_ips,
    )


def is_local_connection(client_ip: Optional[str] = None) -> bool:
    """Return True when the client IP represents localhost."""

    if not client_ip:
        return False

    return client_ip in {"127.0.0.1", "::1", "localhost"}


def is_redirect_uri_allowed(
    redirect_uri: str,
    registered_redirects: Sequence[str],
    allow_list: Sequence[str],
) -> bool:
    """Validate redirect URI against registered URIs and configured allow-list."""

    if redirect_uri in registered_redirects:
        return True

    from urllib.parse import urlparse

    parsed = urlparse(redirect_uri)
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}:
        return True

    for allowed in allow_list:
        if allowed and redirect_uri.startswith(allowed):
            return True

    return False


def extract_api_key_from_headers(
    request_headers: Optional[Dict[str, str]],
) -> Optional[str]:
    """Return API key from headers, handling common capitalizations."""

    if not request_headers:
        return None

    return (
        request_headers.get("x-api-key")
        or request_headers.get("X-API-Key")
        or request_headers.get("X-Api-Key")
    )


def api_key_is_valid(provided_key: str) -> bool:
    """Validate provided API key against env override or stored hash."""

    env_key = get_api_key_from_env()
    if env_key:
        try:
            if secrets.compare_digest(env_key, provided_key):
                return True
        except Exception:
            logger.exception("Failed to compare env API key value securely")
    return verify_api_key(provided_key)


def check_authentication(
    request_headers: Optional[Dict[str, str]] = None,
    client_ip: Optional[str] = None,
    *,
    settings: Optional[AuthSettings] = None,
) -> bool:
    """
    Check if a request is authenticated based on the current settings.
    """

    settings = settings or get_auth_settings()

    if settings.allow_local_bypass and is_local_connection(client_ip):
        logger.debug("Local connection - bypassing authentication")
        return True

    if not settings.auth_enabled:
        return True

    if settings.oauth_enabled and request_headers:
        auth_header = request_headers.get("Authorization") or request_headers.get(
            "authorization"
        )
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
                from ..oauth import get_oauth_manager  # Lazy import to avoid cycles

                oauth_manager = get_oauth_manager()
                if oauth_manager.verify_token(token):
                    logger.debug("OAuth token valid")
                    return True

    api_key = extract_api_key_from_headers(request_headers)
    if api_key and api_key_is_valid(api_key):
        return True

    logger.warning("Authentication failed - no valid credentials provided")
    return False


def check_ip_whitelist(
    client_ip: Optional[str] = None,
    *,
    settings: Optional[AuthSettings] = None,
) -> bool:
    """
    Check if client IP is whitelisted.
    """

    settings = settings or get_auth_settings()

    allowed_ips = settings.allowed_ips
    if not allowed_ips or not allowed_ips[0]:
        return True

    if client_ip in allowed_ips:
        return True

    logger.warning("IP %s not in whitelist", client_ip)
    return False
