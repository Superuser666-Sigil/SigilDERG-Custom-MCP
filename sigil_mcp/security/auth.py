# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

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
    mode: str = "dev"


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
        mode=config.mode,
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

    allowed_ips = [ip for ip in settings.allowed_ips if ip]
    if allowed_ips:
        if not client_ip:
            logger.warning(
                "Authentication failed - client IP missing while whitelist enforced"
            )
            return False
        if client_ip not in allowed_ips:
            logger.warning("Authentication failed - IP %s not allowed", client_ip)
            return False

    if settings.allow_local_bypass and is_local_connection(client_ip):
        if settings.mode == "prod":
            logger.warning(
                "Local authentication bypass accepted while mode=prod for IP %s",
                client_ip,
            )
        else:
            logger.info("Local connection - bypassing authentication for %s", client_ip)
        return True

    if not settings.auth_enabled:
        if settings.mode == "prod":
            logger.warning(
                "Authentication disabled while running in production mode; "
                "all requests will be accepted."
            )
        else:
            logger.debug("Authentication disabled by configuration")
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
                logger.warning("Authentication failed - invalid bearer token")
        else:
            logger.debug("Authentication: no Authorization header present")

    api_key = extract_api_key_from_headers(request_headers)
    if api_key and api_key_is_valid(api_key):
        return True
    elif api_key:
        logger.warning("Authentication failed - provided API key is invalid")

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

    allowed_ips = [ip for ip in settings.allowed_ips if ip]
    if not allowed_ips:
        return True

    if client_ip in allowed_ips:
        return True

    logger.warning("IP %s not in whitelist", client_ip)
    return False
