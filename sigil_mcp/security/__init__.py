# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""Security-related helpers for the Sigil MCP server."""

from .auth import (
    AuthSettings,
    api_key_is_valid,
    check_authentication,
    check_ip_whitelist,
    extract_api_key_from_headers,
    get_auth_settings,
    is_local_connection,
    is_redirect_uri_allowed,
)

__all__ = [
    "AuthSettings",
    "api_key_is_valid",
    "check_authentication",
    "check_ip_whitelist",
    "extract_api_key_from_headers",
    "get_auth_settings",
    "is_local_connection",
    "is_redirect_uri_allowed",
]
