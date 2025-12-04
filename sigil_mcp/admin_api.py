from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .config import get_config
from .server import (
    REPOS,
    _get_index,
    _get_watcher,
    rebuild_index_op,
    build_vector_index_op,
    get_index_stats_op,
)

logger = logging.getLogger("sigil_admin")

_config = get_config()


def _get_admin_cfg() -> Dict[str, Any]:
    return {
        "enabled": _config.admin_enabled,
        "host": _config.admin_host,
        "port": _config.admin_port,
        "api_key": _config.admin_api_key,
        "allowed_ips": _config.admin_allowed_ips,
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

    if not _is_allowed_ip(client_ip):
        logger.warning("Admin access denied from IP %r", client_ip)
        return JSONResponse(
            {"error": "forbidden", "reason": "ip_not_allowed"},
            status_code=403,
        )

    api_key = cfg["api_key"]
    if api_key:
        header_key = (
            request.headers.get("x-admin-key")
            or request.headers.get("X-Admin-Key")
        )
        if header_key != api_key:
            logger.warning("Admin access denied due to invalid API key")
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    return None


async def admin_status(request: Request) -> Response:
    if (resp := await require_admin(request)) is not None:
        return resp

    index = _get_index()
    watcher = _get_watcher()
    cfg = _get_admin_cfg()

    payload: Dict[str, Any] = {
        "admin": {
            "host": cfg["host"],
            "port": cfg["port"],
            "enabled": cfg["enabled"],
        },
        "repos": {name: str(path) for name, path in REPOS.items()},
        "index": {
            "path": str(index.index_path),
            "has_embeddings": bool(getattr(index, "embed_fn", None)),
            "embed_model": getattr(index, "embed_model", None),
        },
        "watcher": {
            "enabled": watcher is not None,
            "watching": list(REPOS.keys()) if watcher is not None else [],
        },
    }
    return JSONResponse(payload)


async def admin_index_rebuild(request: Request) -> Response:
    if (resp := await require_admin(request)) is not None:
        return resp

    body = await request.json() if request.method == "POST" else {}
    repo = body.get("repo")
    force = bool(body.get("force", True))

    try:
        result = rebuild_index_op(repo=repo, force_rebuild=force)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("admin_index_rebuild failed: %s", exc)
        return JSONResponse(
            {"error": "internal_error", "detail": str(exc)},
            status_code=500,
        )


async def admin_index_stats(request: Request) -> Response:
    if (resp := await require_admin(request)) is not None:
        return resp

    repo = request.query_params.get("repo")
    try:
        result = get_index_stats_op(repo=repo)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("admin_index_stats failed: %s", exc)
        return JSONResponse(
            {"error": "internal_error", "detail": str(exc)},
            status_code=500,
        )


async def admin_vector_rebuild(request: Request) -> Response:
    if (resp := await require_admin(request)) is not None:
        return resp

    body = await request.json() if request.method == "POST" else {}
    repo = body.get("repo")
    force = bool(body.get("force", True))
    model = body.get("model", "default")

    try:
        result = build_vector_index_op(repo=repo, force_rebuild=force, model=model)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("admin_vector_rebuild failed: %s", exc)
        return JSONResponse(
            {"error": "internal_error", "detail": str(exc)},
            status_code=500,
        )


async def admin_logs_tail(request: Request) -> Response:
    if (resp := await require_admin(request)) is not None:
        return resp

    n_param = request.query_params.get("n", "200")
    try:
        n = max(1, min(int(n_param), 2000))
    except ValueError:
        n = 200

    log_file_env = getattr(_config, "log_file", None) or None
    if log_file_env is None:
        default_path = Path.home() / ".sigil_mcp_server" / "logs" / "server.log"
        log_path = default_path
    else:
        log_path = Path(str(log_file_env)).expanduser()

    if not log_path.exists():
        # Fallback to try and find where logs might be if not in default location
        # This is a best effort.
        return JSONResponse(
            {"error": "not_found", "detail": f"log file not found: {log_path}"},
            status_code=404,
        )

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
    if (resp := await require_admin(request)) is not None:
        return resp

    raw = _config.config_data
    return JSONResponse(raw)


routes = [
    Route("/admin/status", admin_status, methods=["GET"]),
    Route("/admin/index/rebuild", admin_index_rebuild, methods=["POST"]),
    Route("/admin/index/stats", admin_index_stats, methods=["GET"]),
    Route("/admin/vector/rebuild", admin_vector_rebuild, methods=["POST"]),
    Route("/admin/logs/tail", admin_logs_tail, methods=["GET"]),
    Route("/admin/config", admin_config_view, methods=["GET"]),
]

app = Starlette(debug=False, routes=routes)

# CORS for local development (needed before Stage 2 UI will work)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
