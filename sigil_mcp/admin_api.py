# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from typing import Any, Dict, Optional

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .config import get_config
import sigil_mcp.server as server_state
from .server import (
    REPOS,
    rebuild_index_op,
    build_vector_index_op,
    get_index_stats_op,
    external_mcp_status_op,
    refresh_external_mcp_op,
)

logger = logging.getLogger("sigil_admin")

_config = get_config()

# Semaphore to limit concurrent rebuild operations (prevent database lock contention)
# Only allow one rebuild operation at a time
_rebuild_semaphore = asyncio.Semaphore(1)


def _get_admin_cfg() -> Dict[str, Any]:
    mode_env = os.getenv("SIGIL_MCP_MODE")
    mode = mode_env.lower() if mode_env else _config.mode
    return {
        "enabled": _config.admin_enabled,
        "host": _config.admin_host,
        "port": _config.admin_port,
        "api_key": _config.admin_api_key,
        "require_api_key": _config.admin_require_api_key,
        "allowed_ips": _config.admin_allowed_ips,
        "mode": mode,
    }


def _is_allowed_ip(ip: Optional[str]) -> bool:
    if not ip:
        return False
    cfg = _get_admin_cfg()
    allowed = set(cfg["allowed_ips"] or ["127.0.0.1", "::1"])
    return ip in allowed


async def require_admin(request: Request) -> Optional[JSONResponse]:
    """
    Common gate for all admin endpoints.

    - Enforce local-only IP (admin.allowed_ips)
    - Enforce X-Admin-Key header if admin.api_key is set
    """
    client = request.client
    client_ip = client.host if client else None
    cfg = _get_admin_cfg()

    if not cfg["enabled"]:
        logger.warning("Admin API called but admin.enabled=false")
        return JSONResponse({"error": "admin_disabled"}, status_code=503)

    mode = cfg.get("mode", "dev")

    if mode == "prod" and not cfg.get("api_key"):
        logger.error(
            "Admin API is unavailable in production mode without admin.api_key configured"
        )
        return JSONResponse(
            {"error": "admin_api_key_required", "detail": "Configure admin.api_key"},
            status_code=503,
        )

    if not _is_allowed_ip(client_ip):
        logger.warning("Admin access denied from IP %r", client_ip)
        return JSONResponse(
            {"error": "forbidden", "reason": "ip_not_allowed"},
            status_code=403,
        )

    api_key = cfg.get("api_key")
    require_api_key = True if mode == "prod" else cfg.get("require_api_key", True)
    header_key = (
        request.headers.get("x-admin-key")
        or request.headers.get("X-Admin-Key")
    )

    if require_api_key:
        if not api_key:
            logger.error(
                "Admin API misconfigured: require_api_key=true but no api_key set"
            )
            return JSONResponse(
                {
                    "error": "configuration_error",
                    "reason": "admin_api_key_missing",
                },
                status_code=503,
            )
        if header_key != api_key:
            logger.warning("Admin access denied due to missing/invalid API key")
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    else:
        if api_key and header_key and header_key != api_key:
            logger.warning("Admin access denied due to invalid optional API key")
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    return None


async def admin_status(request: Request) -> Response:
    """Return server/admin/index status or error if admin gate fails."""
    if (resp := await require_admin(request)) is not None:
        return resp

    cfg = _get_admin_cfg()
    repo_map = {name: str(path) for name, path in REPOS.items()}

    # Avoid heavy initialization when no repositories are configured
    if not repo_map:
        payload = {
            "admin": {
                "host": cfg.get("host"),
                "port": cfg.get("port"),
                "enabled": cfg.get("enabled"),
            },
            "repos": {},
            "index": {
                "path": None,
                "has_embeddings": False,
                "embed_model": None,
            },
            "watcher": {"enabled": False, "watching": []},
        }
        return JSONResponse(payload)

    index = getattr(server_state, "_INDEX", None)
    watcher = getattr(server_state, "_WATCHER", None)

    index_payload = {
        "path": str(_config.index_path),
        "has_embeddings": False,
        "embed_model": None,
    }
    if index is not None:
        index_payload = {
            "path": str(index.index_path),
            "has_embeddings": bool(getattr(index, "embed_fn", None)),
            "embed_model": getattr(index, "embed_model", None),
        }

    watcher_payload = {
        "enabled": bool(watcher) and _config.watch_enabled,
        "watching": list(REPOS.keys()) if watcher else [],
    }

    payload: Dict[str, Any] = {
        "admin": {
            "host": cfg["host"],
            "port": cfg["port"],
            "enabled": cfg["enabled"],
        },
        "repos": repo_map,
        "index": index_payload,
        "watcher": watcher_payload,
    }
    return JSONResponse(payload)


async def admin_index_rebuild(request: Request) -> Response:
    """Trigger trigram/symbol rebuild via admin API with lock+retry semantics."""
    if (resp := await require_admin(request)) is not None:
        return resp

    body = await request.json() if request.method == "POST" else {}
    repo = body.get("repo")
    force = bool(body.get("force", True))

    # Use semaphore to ensure only one rebuild operation at a time
    async with _rebuild_semaphore:
        # Retry logic for database locks (file watcher might be active)
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                # Run blocking operation in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,  # Use default executor
                    rebuild_index_op,
                    repo,
                    force,
                )
                return JSONResponse(result)
            except sqlite3.OperationalError as exc:
                error_msg = str(exc)
                if "database is locked" in error_msg.lower():
                    if attempt < max_retries - 1:
                        # Retry after a short delay
                        logger.info(
                            "admin_index_rebuild: database locked, "
                            f"retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        # Final attempt failed
                        logger.warning("admin_index_rebuild: database is locked - %s", exc)
                        return JSONResponse(
                            {
                                "error": "database_locked",
                                "detail": (
                                    "The database is currently locked by another operation. "
                                    "This usually happens when another indexing operation is "
                                    "in progress. Please try again in a few seconds."
                                ),
                                "retry_after": 5,
                            },
                            status_code=503,  # Service Unavailable
                        )
                logger.exception("admin_index_rebuild: SQLite operational error - %s", exc)
                return JSONResponse(
                    {"error": "database_error", "detail": error_msg},
                    status_code=500,
                )
            except Exception as exc:
                logger.exception("admin_index_rebuild failed: %s", exc)
                return JSONResponse(
                    {"error": "internal_error", "detail": str(exc)},
                    status_code=500,
                )
        
        # If we get here, all retries failed (shouldn't happen, but type checker)
        return JSONResponse(
            {"error": "internal_error", "detail": "All retry attempts failed"},
            status_code=500,
        )


async def admin_index_stats(request: Request) -> Response:
    """Return index statistics for all repos or a specific repo."""
    if (resp := await require_admin(request)) is not None:
        return resp

    repo = request.query_params.get("repo")
    try:
        # Run blocking operation in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # Use default executor
            get_index_stats_op,
            repo,
        )
        return JSONResponse(result)
    except sqlite3.OperationalError as exc:
        error_msg = str(exc)
        if "database is locked" in error_msg.lower():
            logger.warning("admin_index_stats: database is locked - %s", exc)
            return JSONResponse(
                {
                    "error": "database_locked",
                    "detail": (
                        "The database is currently locked by another operation. "
                        "Please try again in a few seconds."
                    ),
                    "retry_after": 5,
                },
                status_code=503,  # Service Unavailable
            )
        logger.exception("admin_index_stats: SQLite operational error - %s", exc)
        return JSONResponse(
            {"error": "database_error", "detail": error_msg},
            status_code=500,
        )
    except Exception as exc:
        logger.exception("admin_index_stats failed: %s", exc)
        return JSONResponse(
            {"error": "internal_error", "detail": str(exc)},
            status_code=500,
        )


async def admin_vector_rebuild(request: Request) -> Response:
    """Trigger embedding rebuild for one/all repos with retry on database locks."""
    if (resp := await require_admin(request)) is not None:
        return resp

    body = await request.json() if request.method == "POST" else {}
    repo = body.get("repo")
    force = bool(body.get("force", True))
    model = body.get("model", "default")

    # Use semaphore to ensure only one rebuild operation at a time
    async with _rebuild_semaphore:
        # Retry logic for database locks (file watcher might be active)
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                # Run blocking operation in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,  # Use default executor
                    build_vector_index_op,
                    repo,
                    force,
                    model,
                )
                return JSONResponse(result)
            except sqlite3.OperationalError as exc:
                error_msg = str(exc)
                if "database is locked" in error_msg.lower():
                    if attempt < max_retries - 1:
                        # Retry after a short delay
                        logger.info(
                            "admin_vector_rebuild: database locked, "
                            f"retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        # Final attempt failed
                        logger.warning("admin_vector_rebuild: database is locked - %s", exc)
                        return JSONResponse(
                            {
                                "error": "database_locked",
                                "detail": (
                                    "The database is currently locked by another operation. "
                                    "This usually happens when another indexing operation is "
                                    "in progress. Please try again in a few seconds."
                                ),
                                "retry_after": 5,
                            },
                            status_code=503,  # Service Unavailable
                        )
                logger.exception("admin_vector_rebuild: SQLite operational error - %s", exc)
                return JSONResponse(
                    {"error": "database_error", "detail": error_msg},
                    status_code=500,
                )
            except Exception as exc:
                error_msg = str(exc)
                # Handle database locked errors gracefully
                # (for non-SQLite exceptions that mention it)
                if (
                    "database is locked" in error_msg.lower()
                    or "database_locked" in error_msg.lower()
                ):
                    if attempt < max_retries - 1:
                        logger.info(
                            "admin_vector_rebuild: database locked, "
                            f"retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.warning("admin_vector_rebuild: database is locked - %s", exc)
                        return JSONResponse(
                            {
                                "error": "database_locked",
                                "detail": (
                                    "The database is currently locked by another operation. "
                                    "This usually happens when another indexing operation is "
                                    "in progress. Please try again in a few seconds."
                                ),
                                "retry_after": 5,
                            },
                            status_code=503,  # Service Unavailable
                        )
                logger.exception("admin_vector_rebuild failed: %s", exc)
                return JSONResponse(
                    {"error": "internal_error", "detail": error_msg},
                    status_code=500,
                )
        
        # If we get here, all retries failed (shouldn't happen, but type checker)
        return JSONResponse(
            {"error": "internal_error", "detail": "All retry attempts failed"},
            status_code=500,
        )


async def admin_logs_tail(request: Request) -> Response:
    """Tail the configured log file (or return guidance when not configured)."""
    if (resp := await require_admin(request)) is not None:
        return resp

    n_param = request.query_params.get("n", "200")
    try:
        n = max(1, min(int(n_param), 2000))
    except ValueError:
        n = 200

    # Use the configured log file path
    from .logging_setup import get_log_file_path
    log_path = get_log_file_path(_config.log_file)
    
    if log_path is None:
        # No log file configured
        return JSONResponse({
            "path": "N/A",
            "lines": [
                "No log file configured.",
                "",
                "To enable file logging, configure a log file path in your config.json:",
                '  "server": {',
                '    "log_file": "/path/to/server.log"',
                "  }",
                "",
                "Or set the environment variable:",
                "  export SIGIL_MCP_LOG_FILE=/path/to/server.log",
            ],
            "note": "Logs are currently being written to stdout/stderr, not a file."
        })

    if not log_path.exists():
        # Log file doesn't exist - server may be logging to stdout/stderr
        # Return a helpful message instead of an error
        return JSONResponse({
            "path": str(log_path),
            "lines": [
                f"Log file not found: {log_path}",
                "",
                "The server may be logging to stdout/stderr instead of a file.",
                "To enable file logging, configure a log file path in your config.json:",
                '  "server": {',
                '    "log_file": "/path/to/server.log"',
                "  }",
                "",
                "Or set the environment variable:",
                "  export SIGIL_MCP_LOG_FILE=/path/to/server.log",
            ],
            "note": "Logs are currently being written to stdout/stderr, not a file."
        })

    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as exc:
        logger.exception("Failed to read log file: %s", exc)
        return JSONResponse(
            {"error": "internal_error", "detail": str(exc)},
            status_code=500,
        )

    tail = lines[-n:]
    return JSONResponse({"path": str(log_path), "lines": tail})


async def admin_config_view(request: Request) -> Response:
    """Expose the raw config (read-only) for debugging."""
    if (resp := await require_admin(request)) is not None:
        return resp

    raw = _config.config_data
    return JSONResponse(raw)


async def admin_mcp_status(request: Request) -> Response:
    """Return status of external MCP aggregation."""
    if (resp := await require_admin(request)) is not None:
        return resp
    return JSONResponse(external_mcp_status_op())


async def admin_mcp_refresh(request: Request) -> Response:
    """Force refresh of external MCP servers and tool registration."""
    if (resp := await require_admin(request)) is not None:
        return resp
    try:
        status = refresh_external_mcp_op()
        return JSONResponse(status)
    except Exception as exc:
        logger.exception("admin_mcp_refresh failed: %s", exc)
        return JSONResponse({"error": "refresh_failed", "detail": str(exc)}, status_code=500)


async def root(request: Request) -> Response:
    """Root endpoint that lists available Admin API endpoints."""
    return JSONResponse({
        "service": "Sigil MCP Admin API",
        "version": "0.4.0",
        "endpoints": {
            "GET /admin/status": "Server status, repositories, index info, watcher status",
            "GET /admin/index/stats": "Get index statistics (all repos or specific repo)",
            "POST /admin/index/rebuild": (
                "Rebuild trigram/symbol index (all repos or specific repo)"
            ),
            "POST /admin/vector/rebuild": "Rebuild vector embeddings index",
            "GET /admin/logs/tail": "Get last N lines from server log file (query param: ?lines=N)",
            "GET /admin/config": "View current configuration (read-only)",
            "GET /admin/mcp/status": "External MCP aggregation status",
            "POST /admin/mcp/refresh": "Refresh/reload external MCP servers and tools",
        },
        "documentation": "See docs/RUNBOOK.md for complete Admin API documentation",
    })


routes = [
    Route("/admin", root, methods=["GET"]),  # Admin API root (changed from "/" to avoid conflicts)
    Route("/admin/status", admin_status, methods=["GET"]),
    Route("/admin/index/rebuild", admin_index_rebuild, methods=["POST"]),
    Route("/admin/index/stats", admin_index_stats, methods=["GET"]),
    Route("/admin/vector/rebuild", admin_vector_rebuild, methods=["POST"]),
    Route("/admin/logs/tail", admin_logs_tail, methods=["GET"]),
    Route("/admin/config", admin_config_view, methods=["GET"]),
    Route("/admin/mcp/status", admin_mcp_status, methods=["GET"]),
    Route("/admin/mcp/refresh", admin_mcp_refresh, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes)

# CORS for local development (needed before Stage 2 UI will work)
# Never permit wildcard origins; tighten defaults for production.
_base_admin_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_prod_admin_origins = [
    origin for origin in _base_admin_origins if "localhost" not in origin
] or _base_admin_origins
_allowed_origins = _base_admin_origins if _config.mode == "dev" else _prod_admin_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
