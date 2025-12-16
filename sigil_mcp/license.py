"""Simple license gating helpers and decorator.

This module provides a small, testable gating mechanism used to protect
pro-only features. The implementation is intentionally lightweight and
configurable via environment variable `SIGIL_MCP_PRO_ACTIVE` for tests and
simple deployments. In the future this can be extended to verify JWTs or
call a license server.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import wraps

logger = logging.getLogger(__name__)


def is_pro_active() -> bool:
    """Return True when a Pro license is active.

    By default this checks the environment variable `SIGIL_MCP_PRO_ACTIVE`.
    Tests can toggle this by setting the environment variable via monkeypatch.
    """
    try:
        val = os.getenv("SIGIL_MCP_PRO_ACTIVE", "").strip()
        return val == "1" or val.lower() in ("true", "yes", "on")
    except Exception:
        return False


def require_pro_feature(feature_name: str) -> Callable:
    """Decorator that raises PermissionError when a Pro license is not active.

    Usage:
        @require_pro_feature("advanced-embeddings")
        def create_advanced_index(...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_pro_active():
                logger.warning("Blocked access to Pro feature: %s", feature_name)
                raise PermissionError(f"{feature_name} requires a Pro license")
            return func(*args, **kwargs)

        return wrapper

    return decorator
