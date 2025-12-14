# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

import sigil_mcp.server as server_state

from .config import get_config, load_config, save_config
from .server import (
    REPOS,
    build_vector_index_op,
    external_mcp_status_op,
    get_index_stats_op,
    rebuild_index_op,
    refresh_external_mcp_op,
)

logger = logging.getLogger("sigil_admin")

_config = get_config()

# Semaphore to limit concurrent rebuild operations (prevent database lock contention)
# Only allow one rebuild operation at a time
_rebuild_semaphore = asyncio.Semaphore(1)


def _get_admin_cfg() -> dict[str, Any]:
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


def _is_allowed_ip(ip: str | None) -> bool:
    if not ip:
        return False
    cfg = _get_admin_cfg()
    allowed = set(cfg["allowed_ips"] or ["127.0.0.1", "::1"])
    return ip in allowed


async def require_admin(request: Request) -> JSONResponse | None:
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
                "mode": cfg.get("mode", "dev"),
            },
            "repos": {},
            "index": {
                "path": None,
                # embeddings_configured indicates the config flag; embeddings_ready indicates the runtime function is available
                "embeddings_configured": bool(_config.embeddings_enabled),
                "embeddings_ready": False,
                "has_embeddings": False,
                "embed_model": None,
                "trigram_backend": None,
                "trigram_path": None,
                "vector_backend": None,
                "vector_path": None,
            },
            "watcher": {"enabled": False, "watching": []},
        }
        return JSONResponse(payload)

    index = getattr(server_state, "_INDEX", None)
    watcher = getattr(server_state, "_WATCHER", None)

    index_payload = {
        "path": str(_config.index_path),
        # embeddings_configured reflects configuration; embeddings_ready reflects runtime availability
        "embeddings_configured": bool(_config.embeddings_enabled),
        "embeddings_ready": False,
        "has_embeddings": False,
        "embed_model": None,
        "trigram_backend": None,
        "trigram_path": None,
        "vector_backend": None,
        "vector_path": None,
    }
    if index is not None:
        trig_backend = getattr(index, "_trigram_backend", None)
        trig_path = None
        try:
            trig_path = str(index.index_path / "trigrams.rocksdb")
        except Exception:
            trig_path = None
        vec_backend = "lancedb" if getattr(index, "lance_db", None) else None
        vec_path = None
        try:
            vec_path = str(getattr(index, "lance_db_path", None) or "")
        except Exception:
            vec_path = None
        index_payload = {
            "path": str(index.index_path),
            "embeddings_configured": bool(_config.embeddings_enabled),
            "embeddings_ready": bool(getattr(index, "embed_fn", None)),
            "has_embeddings": bool(getattr(index, "embed_fn", None)),
            "embed_model": getattr(index, "embed_model", None),
            "trigram_backend": trig_backend,
            "trigram_path": trig_path,
            "vector_backend": vec_backend,
            "vector_path": vec_path,
        }

    watcher_payload = {
        "enabled": bool(watcher) and _config.watch_enabled,
        "watching": list(REPOS.keys()) if watcher else [],
    }

    payload: dict[str, Any] = {
        "admin": {
            "host": cfg["host"],
            "port": cfg["port"],
            "enabled": cfg["enabled"],
            "mode": cfg.get("mode", "dev"),
        },
        "repos": repo_map,
        "index": index_payload,
        "watcher": watcher_payload,
    }
    return JSONResponse(payload)


async def admin_get_repo(request: Request) -> Response:
    if (resp := await require_admin(request)) is not None:
        return resp

    name = request.path_params.get("name")
    if not name:
        return JSONResponse({"error": "missing_repo_name"}, status_code=400)

    repo_opts = server_state.REPO_OPTIONS.get(name)
    if not repo_opts:
        # fallback: repo may exist in REPOS without options
        if name in REPOS:
            # attempt to read per-repo embeddings.include_solution from repos DB
            include_solution = None
            try:
                index = getattr(server_state, "_INDEX", None) or server_state._get_index()
                cur = index.repos_db.cursor()
                cur.execute("SELECT embeddings_include_solution FROM repos WHERE name = ?", (name,))
                row = cur.fetchone()
                if row:
                    include_solution = bool(row[0]) if row[0] is not None else None
            except Exception:
                include_solution = None
            payload = {"name": name, "path": str(REPOS[name]), "respect_gitignore": True}
            if include_solution is not None:
                payload["embeddings_include_solution"] = include_solution
            return JSONResponse(payload)
        return JSONResponse({"error": "repo_not_found"}, status_code=404)

    payload = {
        "name": name,
        "path": str(repo_opts.get("path")),
        "respect_gitignore": bool(repo_opts.get("respect_gitignore", True)),
        "ignore_patterns": list(repo_opts.get("ignore_patterns", []) or []),
    }
    if "embeddings_include_solution" in repo_opts:
        payload["embeddings_include_solution"] = bool(repo_opts.get("embeddings_include_solution"))
    else:
        # If not present in runtime options, attempt to read from repos DB
        try:
            index = getattr(server_state, "_INDEX", None) or server_state._get_index()
            cur = index.repos_db.cursor()
            cur.execute("SELECT embeddings_include_solution FROM repos WHERE name = ?", (name,))
            row = cur.fetchone()
            if row and row[0] is not None:
                payload["embeddings_include_solution"] = bool(row[0])
        except Exception:
            pass

    return JSONResponse(payload)


async def admin_set_repo_gitignore(request: Request) -> Response:
    if (resp := await require_admin(request)) is not None:
        return resp

    name = request.path_params.get("name")
    if not name:
        return JSONResponse({"error": "missing_repo_name"}, status_code=400)

    body = await request.json() if request.method == "POST" else {}
    if "respect_gitignore" not in body:
        return JSONResponse({"error": "missing_field", "field": "respect_gitignore"}, status_code=400)

    val = bool(body.get("respect_gitignore", True))

    # Optional: per-repo ignore_patterns provided via admin API
    ignore_patterns_body = body.get("ignore_patterns", None)
    if ignore_patterns_body is not None:
        try:
            ignore_patterns_val = list(ignore_patterns_body) if isinstance(ignore_patterns_body, (list, tuple)) else []
        except Exception:
            ignore_patterns_val = []
    else:
        ignore_patterns_val = None

    # Optional: per-repo embeddings.include_solution flag
    include_sol_body = body.get("embeddings_include_solution", None)
    if include_sol_body is not None:
        include_sol_val = bool(include_sol_body)
    else:
        include_sol_val = None

    if name not in server_state.REPO_OPTIONS and name not in REPOS:
        return JSONResponse({"error": "repo_not_found"}, status_code=404)

    # Ensure REPO_OPTIONS entry exists
    if name not in server_state.REPO_OPTIONS:
        server_state.REPO_OPTIONS[name] = {"path": REPOS[name], "respect_gitignore": True}

    server_state.REPO_OPTIONS[name]["respect_gitignore"] = val
    if include_sol_val is not None:
        server_state.REPO_OPTIONS[name]["embeddings_include_solution"] = include_sol_val
    if ignore_patterns_val is not None:
        server_state.REPO_OPTIONS[name]["ignore_patterns"] = list(ignore_patterns_val)

    # Update watcher runtime if present
    watcher = getattr(server_state, "_WATCHER", None)
    repo_path = Path(server_state.REPO_OPTIONS[name]["path"]) if server_state.REPO_OPTIONS[name].get("path") else Path(REPOS[name])
    try:
        if watcher:
            # restart watching for this repo to apply new gitignore behaviour
            watcher.unwatch_repository(name)
            watcher.watch_repository(name, repo_path, honor_gitignore=val, repo_ignore_patterns=server_state.REPO_OPTIONS[name].get("ignore_patterns"))
    except Exception:
        # non-fatal; continue
        pass

    # Persist change to config.json if possible
    persisted = False
    try:
        # Ensure repositories mapping exists in config_data
        repos = _config.config_data.setdefault("repositories", {})
        # Normalize to dict form with path and respect_gitignore
        try:
            repo_str_path = str(repo_path)
        except Exception:
            repo_str_path = str(REPOS.get(name, repo_path))

        repos[name] = {"path": repo_str_path, "respect_gitignore": bool(val)}
        if ignore_patterns_val is not None:
            repos[name]["ignore_patterns"] = list(ignore_patterns_val)
        if include_sol_val is not None:
            repos[name]["embeddings_include_solution"] = bool(include_sol_val)
        save_config(_config)
        persisted = True
    except Exception:
        logger.exception("Failed to persist repo gitignore setting for %s", name)

    return JSONResponse({"name": name, "respect_gitignore": val, "persisted": persisted})


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


async def admin_index_file_rebuild(request: Request) -> Response:
    """Trigger reindex for a single file (repo + path) via admin API."""
    if (resp := await require_admin(request)) is not None:
        return resp

    body = await request.json() if request.method == "POST" else {}
    repo = body.get("repo")
    path = body.get("path")

    if not repo or not path:
        return JSONResponse({"error": "invalid_request", "detail": "Missing repo or path"}, status_code=400)

    # Ensure repo configured
    if repo not in REPOS:
        return JSONResponse({"error": "unknown_repo", "detail": f"Repository {repo} not configured"}, status_code=404)

    # Use semaphore to avoid concurrent DB writes
    async with _rebuild_semaphore:
        try:
            loop = asyncio.get_event_loop()
            repo_path = REPOS[repo]
            # Run blocking index_file in executor (blocking helper handles retries)
            result = await loop.run_in_executor(
                None,
                lambda: _run_index_file(repo, repo_path, path),
            )

            # `_run_index_file` now returns a structured dict {"ok": bool, "indexed": bool, "skipped": bool, "error": str}
            if isinstance(result, dict):
                if result.get("ok"):
                    return JSONResponse({"success": True, "indexed": bool(result.get("indexed", False)), "skipped": bool(result.get("skipped", False))})
                else:
                    # Return detailed error information when available
                    detail = result.get("error", "failed to index file")
                    if "database is locked" in str(detail).lower():
                        return JSONResponse({"error": "database_locked", "detail": detail}, status_code=503)
                    return JSONResponse({"error": "index_failed", "detail": detail}, status_code=500)

            # Fallback for older boolean-style return values
            if result is True:
                return JSONResponse({"success": True, "indexed": True})
            if result is False:
                return JSONResponse({"success": True, "indexed": False, "skipped": True})

            # Unexpected return
            return JSONResponse({"success": False, "detail": "unexpected_result"}, status_code=500)
        except sqlite3.OperationalError as exc:
            error_msg = str(exc)
            if "database is locked" in error_msg.lower():
                logger.warning("admin_index_file_rebuild: database is locked - %s", exc)
                return JSONResponse({
                    "error": "database_locked",
                    "detail": "Database locked; try again",
                }, status_code=503)
            logger.exception("admin_index_file_rebuild: SQLite error - %s", exc)
            return JSONResponse({"error": "database_error", "detail": error_msg}, status_code=500)
        except Exception as exc:
            logger.exception("admin_index_file_rebuild failed: %s", exc)
            return JSONResponse({"error": "internal_error", "detail": str(exc)}, status_code=500)


def _run_index_file(repo: str, repo_path: str, path: str) -> dict:
    """Blocking helper to call index.index_file with proper Path conversion.

    Returns a structured dict with keys:
      - ok: bool
      - indexed: bool (True if indexing work performed)
      - skipped: bool (True if no work was needed)
      - error: optional error message
    This helper will retry briefly on SQLite 'database is locked' errors to
    reduce flakiness from concurrent watcher activity.
    """
    from pathlib import Path as _Path
    max_retries = 4
    base_delay = 0.5

    repo_root = _Path(repo_path)
    file_path = _Path(path)
    if not file_path.is_absolute():
        file_path = repo_root / file_path

    try:
        index = getattr(server_state, "_INDEX", None)
        if index is None:
            # Ensure index is initialized
            index = server_state._get_index()
    except Exception as e:
        logger.exception("_run_index_file: failed to get index instance: %s", e)
        return {"ok": False, "error": f"index_unavailable: {e}"}

    for attempt in range(1, max_retries + 1):
        try:
            ok = index.index_file(repo, repo_root, file_path)
            # index.index_file returns True when work was done, False when skipped
            return {"ok": True, "indexed": bool(ok), "skipped": not bool(ok)}
        except sqlite3.OperationalError as exc:
            msg = str(exc)
            if "database is locked" in msg.lower():
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.info("_run_index_file: database locked, retrying in %.1fs (attempt %d/%d)", delay, attempt, max_retries)
                    import time

                    time.sleep(delay)
                    continue
                logger.exception("_run_index_file: database is locked on final attempt - %s", exc)
                return {"ok": False, "error": msg}
            # Not a lock error - propagate as structured failure
            logger.exception("_run_index_file: sqlite error indexing %s:%s - %s", repo, path, exc)
            return {"ok": False, "error": msg}
        except Exception as exc:
            logger.exception("_run_index_file failed for %s:%s", repo, path)
            return {"ok": False, "error": str(exc)}


async def admin_index_stale(request: Request) -> Response:
    """Return per-repo LanceDB path, size, and list of stale documents needing vector update."""
    if (resp := await require_admin(request)) is not None:
        return resp

    repo = request.query_params.get("repo")

    try:
        index = getattr(server_state, "_INDEX", None) or server_state._get_index()
    except Exception:
        logger.exception("admin_index_stale: failed to get index instance")
        return JSONResponse({"error": "internal_error", "detail": "Index unavailable"}, status_code=500)

    results: dict[str, dict[str, object]] = {}

    def _compute_lancedb_info(path) -> dict[str, object]:
        info = {"lance_db_path": str(path), "lance_db_size": 0}
        try:
            total = 0
            if path and os.path.exists(path):
                for root, _, files in os.walk(path):
                    for f in files:
                        try:
                            fp = os.path.join(root, f)
                            total += os.path.getsize(fp)
                        except Exception:
                            continue
            info["lance_db_size"] = total
        except Exception:
            logger.debug("Failed to compute lance db size for %s", path, exc_info=True)
        return info

    try:
        cur = index.repos_db.cursor()
        repos = [repo] if repo else list(REPOS.keys())
        for repo_name in repos:
            repo_entry = {"lance_db_path": None, "lance_db_size": 0, "vectors_stale": 0, "stale_documents": []}
            try:
                # Use per-repo LanceDB path when available
                try:
                    repo_path = getattr(index, "_get_repo_lance_path", None)
                    if callable(repo_path):
                        lpath = index._get_repo_lance_path(repo_name)
                        repo_entry.update(_compute_lancedb_info(lpath))
                    else:
                        repo_entry.update(_compute_lancedb_info(index.lance_db_path))
                except Exception:
                    repo_entry.update(_compute_lancedb_info(index.lance_db_path))

                # Attempt to compute per-repo vector counts using per-repo tables when possible
                repo_entry["lance_db_shared"] = False
                repo_entry["lance_db_total_vectors"] = 0
                repo_entry["vectors_in_repo"] = 0
                repo_entry["lance_db_size_estimate_bytes"] = 0
                try:
                    try:
                        repo_db, repo_table = index._get_repo_lance_and_vectors(repo_name)
                        if repo_table is not None:
                            # Prefer fast count
                            try:
                                repo_entry["vectors_in_repo"] = int(repo_table.count_rows())
                            except Exception:
                                # Try to enumerate as fallback
                                if hasattr(repo_table, "to_list"):
                                    try:
                                        rows = repo_table.to_list()
                                        repo_entry["vectors_in_repo"] = len(rows)
                                    except Exception:
                                        repo_entry["vectors_in_repo"] = 0
                        # If we also have a global table, report its totals for context
                        if getattr(index, "vectors", None) is not None:
                            try:
                                repo_entry["lance_db_total_vectors"] = int(index.vectors.count_rows())
                            except Exception:
                                repo_entry["lance_db_total_vectors"] = 0
                    except Exception:
                        logger.debug("Per-repo lance inspection failed for %s", repo_name, exc_info=True)
                except Exception:
                    logger.debug("Vector table inspection failed", exc_info=True)

                # repo id
                cur.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
                row = cur.fetchone()
                if row:
                    repo_id = row[0]
                    cur.execute(
                        "SELECT path, vector_index_error FROM documents WHERE repo_id = ? AND (vector_index_error IS NOT NULL OR vector_indexed_at IS NULL)",
                        (repo_id,),
                    )
                    docs = cur.fetchall()
                    repo_entry["vectors_stale"] = len(docs)
                    repo_entry["stale_documents"] = [
                        {"path": r[0], "error": r[1]} for r in docs
                    ]
                else:
                    repo_entry["vectors_stale"] = 0
                    repo_entry["stale_documents"] = []
            except Exception:
                logger.exception("Failed to compute stale docs for %s", repo_name)
            results[repo_name] = repo_entry

        return JSONResponse({"success": True, "repos": results})
    except Exception as exc:
        logger.exception("admin_index_stale failed: %s", exc)
        return JSONResponse({"error": "internal_error", "detail": str(exc)}, status_code=500)


async def admin_hardwrap_report(request: Request) -> Response:
    """Return a report of files that triggered many hard-wrap splits."""
    if (resp := await require_admin(request)) is not None:
        return resp

    repo = request.query_params.get("repo")
    min_hardwraps = int(request.query_params.get("min_hardwraps", "5"))
    top_n = int(request.query_params.get("top_n", "200"))

    try:
        index = getattr(server_state, "_INDEX", None) or server_state._get_index()
    except Exception:
        logger.exception("admin_hardwrap_report: index unavailable")
        return JSONResponse({"error": "index_unavailable"}, status_code=500)

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, lambda: index.generate_hardwrap_report(repo=repo, top_n=top_n))
        # Filter by min_hardwraps
        filtered = [r for r in results if r.get("oversized_count", 0) >= min_hardwraps]
        return JSONResponse({"success": True, "report": filtered})
    except Exception as exc:
        logger.exception("admin_hardwrap_report failed: %s", exc)
        return JSONResponse({"error": "internal_error", "detail": str(exc)}, status_code=500)


async def admin_set_repo_include_solution(request: Request) -> Response:
    """Set per-repo embeddings.include_solution option via Admin API."""
    if (resp := await require_admin(request)) is not None:
        return resp

    name = request.path_params.get("name")
    if not name:
        return JSONResponse({"error": "missing_repo_name"}, status_code=400)

    body = await request.json() if request.method == "POST" else {}
    if "embeddings_include_solution" not in body:
        return JSONResponse({"error": "missing_field", "field": "embeddings_include_solution"}, status_code=400)

    val = bool(body.get("embeddings_include_solution"))

    if name not in server_state.REPO_OPTIONS and name not in REPOS:
        return JSONResponse({"error": "repo_not_found"}, status_code=404)

    # Ensure REPO_OPTIONS entry exists
    if name not in server_state.REPO_OPTIONS:
        server_state.REPO_OPTIONS[name] = {"path": REPOS[name], "respect_gitignore": True}

    server_state.REPO_OPTIONS[name]["embeddings_include_solution"] = val

    # Persist to repos DB (best-effort)
    try:
        index = getattr(server_state, "_INDEX", None)
        if index is None:
            index = server_state._get_index()
        cur = index.repos_db.cursor()
        cur.execute("INSERT OR IGNORE INTO repos (name, path, indexed_at) VALUES (?, ?, ?)", (name, server_state.REPO_OPTIONS[name]["path"], datetime.now().isoformat()))
        cur.execute("UPDATE repos SET embeddings_include_solution = ? WHERE name = ?", (1 if val else 0, name))
        index.repos_db.commit()
    except Exception:
        logger.exception("Failed to persist per-repo embeddings_include_solution to DB for %s", name)

    # Persist change to config.json if possible
    persisted = False
    try:
        repos = _config.config_data.setdefault("repositories", {})
        repo_str_path = str(server_state.REPO_OPTIONS[name]["path"])
        entry = repos.get(name)
        if not entry or isinstance(entry, str):
            repos[name] = {"path": repo_str_path, "respect_gitignore": bool(server_state.REPO_OPTIONS[name].get("respect_gitignore", True)), "embeddings_include_solution": val}
        else:
            repos[name]["embeddings_include_solution"] = val
        save_config(_config)
        persisted = True
    except Exception:
        logger.exception("Failed to persist repo embeddings_include_solution to config.json for %s", name)

    return JSONResponse({"name": name, "embeddings_include_solution": val, "persisted": persisted})


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
    # Hide secrets (but keep structure)
    try:
        raw_safe = dict(raw)
        emb = dict(raw_safe.get("embeddings", {}))
        if "api_key" in emb:
            emb["api_key"] = "***"
        raw_safe["embeddings"] = emb
    except Exception:
        raw_safe = raw
    return JSONResponse(raw_safe)


async def admin_config_update(request: Request) -> Response:
    """Persist a new config.json payload (dev mode only)."""
    if (resp := await require_admin(request)) is not None:
        return resp

    global _config

    if _config.mode != "dev":
        return JSONResponse(
            {"error": "forbidden", "detail": "Config editing allowed only in dev mode"},
            status_code=403,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "invalid_json", "detail": "Body must be valid JSON object"},
            status_code=400,
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "invalid_payload", "detail": "Config payload must be an object"},
            status_code=400,
        )

    try:
        _config.config_data = body
        out_path = save_config(_config)
        _config = load_config(out_path)
        return JSONResponse(
            {"success": True, "path": str(out_path), "config": _config.config_data}
        )
    except Exception as exc:
        logger.exception("Failed to persist config.json via admin API")
        return JSONResponse(
            {"error": "config_save_failed", "detail": str(exc)}, status_code=500
        )


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
            "POST /admin/config": "Update configuration (dev mode only)",
            "GET /admin/mcp/status": "External MCP aggregation status",
            "POST /admin/mcp/refresh": "Refresh/reload external MCP servers and tools",
        },
        "documentation": "See docs/RUNBOOK.md for complete Admin API documentation",
    })


routes = [
    Route("/admin", root, methods=["GET"]),  # Admin API root (changed from "/" to avoid conflicts)
    Route("/admin/status", admin_status, methods=["GET"]),
    Route("/admin/index/rebuild", admin_index_rebuild, methods=["POST"]),
    Route("/admin/index/file/rebuild", admin_index_file_rebuild, methods=["POST"]),
    Route("/admin/index/stale", admin_index_stale, methods=["GET"]),
    Route("/admin/index/stats", admin_index_stats, methods=["GET"]),
    Route("/admin/vector/rebuild", admin_vector_rebuild, methods=["POST"]),
    Route("/admin/logs/tail", admin_logs_tail, methods=["GET"]),
    Route("/admin/config", admin_config_view, methods=["GET"]),
    Route("/admin/config", admin_config_update, methods=["POST"]),
    Route("/admin/mcp/status", admin_mcp_status, methods=["GET"]),
    Route("/admin/mcp/refresh", admin_mcp_refresh, methods=["POST"]),
    Route("/admin/repo/{name}", admin_get_repo, methods=["GET"]),
    Route("/admin/repo/{name}/gitignore", admin_set_repo_gitignore, methods=["POST"]),
    Route("/admin/repo/{name}/embeddings_include_solution", admin_set_repo_include_solution, methods=["POST"]),
    Route("/admin/report/hardwraps", admin_hardwrap_report, methods=["GET"]),
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
