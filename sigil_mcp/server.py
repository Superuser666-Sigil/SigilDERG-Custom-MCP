# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, cast
from urllib.parse import urlencode
import numpy as np
import os
import asyncio
import subprocess

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, HTMLResponse
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import UploadFile
from starlette.applications import Starlette
from starlette.routing import Mount
from pathlib import PurePosixPath
from fastmcp.server.http import create_sse_app
from .indexer import SigilIndex
from .auth import initialize_api_key
from .oauth import get_oauth_manager
from .config import get_config
from .watcher import FileWatchManager, WATCHDOG_AVAILABLE
from .app_factory import build_mcp_app
from .mcp_client import (
    MCPClientManager,
    ExternalMCPConfigError,
    set_global_manager,
    get_global_manager,
)
from .admin_ui import start_admin_ui, stop_admin_ui
from . import mcp_installer
from .security import (
    AuthSettings,
    check_authentication as _security_check_authentication,
    check_ip_whitelist as _security_check_ip_whitelist,
    is_local_connection as _security_is_local_connection,
    is_redirect_uri_allowed as _security_is_redirect_uri_allowed,
)
from .middleware.header_logging import HeaderLoggingASGIMiddleware

# Import Admin API app (conditional to avoid circular imports)
_admin_app = None


def _get_admin_app():
    """Lazy import of Admin API app to avoid circular dependencies."""
    global _admin_app
    if _admin_app is None:
        from .admin_api import app
        _admin_app = app
    return _admin_app


# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------

def get_form_value(value: Union[str, UploadFile, None]) -> Optional[str]:
    """
    Extract string value from form data.
    Starlette form() can return str | UploadFile, but OAuth params are always strings.
    
    Args:
        value: Form value which might be str, UploadFile, or None
        
    Returns:
        String value or None
    """
    if isinstance(value, str):
        return value
    if isinstance(value, UploadFile):
        # This shouldn't happen for OAuth params, but handle gracefully
        return None
    return None


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

config = get_config()
RUN_MODE = config.mode

logger = logging.getLogger("sigil_repos_mcp")
mcp = build_mcp_app(config)
_MCP_CLIENT_MANAGER: Optional[MCPClientManager] = None

# Track readiness across subsystems
READINESS: Dict[str, bool] = {
    "config": True,
    "index": False,
    "embeddings": not config.embeddings_enabled,  # ready when disabled
    "dependencies": False,
    "watcher": not config.watch_enabled,
}


def _log_configuration_summary() -> None:
    """Emit a concise configuration summary at startup for operators."""

    logger.info(
        "Sigil MCP starting (mode=%s, auth_enabled=%s, allow_local_bypass=%s, "
        "oauth_enabled=%s)",
        RUN_MODE,
        config.auth_enabled,
        config.allow_local_bypass,
        config.oauth_enabled,
    )
    logger.info(
        "Embeddings: enabled=%s provider=%s model=%s lance_dir=%s",
        config.embeddings_enabled,
        config.embeddings_provider,
        config.embeddings_model,
        config.lance_dir,
    )
    logger.info(
        "Admin API: enabled=%s host=%s port=%s require_api_key=%s",
        config.admin_enabled,
        config.admin_host,
        config.admin_port,
        config.admin_require_api_key,
    )


_log_configuration_summary()


def _check_dependencies() -> None:
    """Validate native/optional dependencies and set readiness flags."""
    global READINESS
    deps_ok = True
    # Trigram backend (rocksdict) is required
    try:
        import rocksdict  # noqa: F401
    except Exception as exc:
        deps_ok = False
        logger.error(
            "Missing required dependency 'rocksdict' (or RocksDB). Install it before starting. %s",
            exc,
        )
        raise RuntimeError("rocksdict dependency missing") from exc

    # Vector store (LanceDB/pyarrow) required when embeddings enabled
    if config.embeddings_enabled:
        try:
            import lancedb  # noqa: F401
        except Exception as exc:
            deps_ok = False
            logger.error(
                "Embeddings enabled but LanceDB is unavailable. Install with `pip install .[lancedb]`."
            )
            raise RuntimeError("lancedb dependency missing") from exc
        try:
            import pyarrow  # noqa: F401
        except Exception as exc:
            deps_ok = False
            logger.error(
                "Embeddings enabled but pyarrow is unavailable. Install with `pip install .[lancedb]`."
            )
            raise RuntimeError("pyarrow dependency missing") from exc

    # Watcher is optional; mark readiness but do not fail startup
    if config.watch_enabled and not WATCHDOG_AVAILABLE:
        deps_ok = False
        logger.warning(
            "File watching is enabled but watchdog is not installed. Install with `pip install watchdog`."
        )
    READINESS["dependencies"] = deps_ok


_check_dependencies()

# --------------------------------------------------------------------
# Security Configuration
# --------------------------------------------------------------------

# Load from config (falls back to environment variables)
AUTH_ENABLED = config.auth_enabled
OAUTH_ENABLED = config.oauth_enabled
ALLOW_LOCAL_BYPASS = config.allow_local_bypass
ALLOWED_IPS = config.allowed_ips
if RUN_MODE == "prod" and ALLOW_LOCAL_BYPASS:
    logger.warning(
        "Local authentication bypass is enabled in production mode - disable "
        "authentication.allow_local_bypass for secure deployments."
    )

# --------------------------------------------------------------------
# MCP server
# --------------------------------------------------------------------

# Helper: safe decorator wrapper for attaching metadata to MCP tools.

# We try to call `mcp.tool` with the provided kwargs; if the installed
# FastMCP decorator doesn't accept these kwargs, we fall back to the
# simple decorator and attach metadata attributes to the wrapped function
# to help downstream clients that inspect function attributes.


def _safe_tool_decorator(**meta_kwargs):
    def _decorator(func):
        dec = getattr(mcp, "tool", None)
        if dec is None:
            return func

        # Prefer passing metadata to the decorator; fall back if not supported.
        try:
            wrapped = dec(**meta_kwargs)(func)
        except TypeError:
            wrapped = dec()(func)

        # Attach conservative metadata attributes on the wrapped function
        # so that clients that inspect function objects can see hints.
        attr_map = {
            "annotations": "__mcp_annotations__",
            "title": "__mcp_title__",
            "description": "__mcp_description__",
            "inputSchema": "__mcp_input_schema__",
        }
        for key, attr_name in attr_map.items():
            if key in meta_kwargs:
                try:
                    setattr(wrapped, attr_name, meta_kwargs[key])
                except Exception:
                    # Best-effort only; do not fail registration
                    pass

        return wrapped

    return _decorator


# --------------------------------------------------------------------
# Authentication Middleware
# --------------------------------------------------------------------

def _get_auth_settings_override() -> AuthSettings:
    """
    Construct an AuthSettings object from the server-level configuration flags.

    Tests often monkeypatch AUTH_ENABLED/OAUTH_ENABLED/etc., so we read the globals
    each time rather than caching the object.
    """

    allowed_ips: Sequence[str] = tuple(ALLOWED_IPS or [])
    return AuthSettings(
        auth_enabled=AUTH_ENABLED,
        oauth_enabled=OAUTH_ENABLED,
        allow_local_bypass=ALLOW_LOCAL_BYPASS,
        allowed_ips=allowed_ips,
        mode=RUN_MODE,
    )


def is_local_connection(client_ip: Optional[str] = None) -> bool:
    """Compatibility wrapper around sigil_mcp.security.is_local_connection."""

    return _security_is_local_connection(client_ip)

def _initialize_external_mcp():
    """Initialize external MCP servers and register their tools."""
    global _MCP_CLIENT_MANAGER
    if _MCP_CLIENT_MANAGER is not None:
        return
    servers = config.external_mcp_servers
    if not servers:
        return
    if config.external_mcp_auto_install:
        mcp_installer.auto_install(servers)
    try:
        manager = MCPClientManager(servers, logger=logger)
    except ExternalMCPConfigError as exc:
        logger.warning("External MCP configuration invalid: %s", exc)
        return

    try:
        asyncio.run(manager.register_with_fastmcp(mcp, tool_decorator=_safe_tool_decorator))
        _MCP_CLIENT_MANAGER = manager
        set_global_manager(manager)
        logger.info(
            "External MCP servers registered: %d tools", len(manager.list_registered_tools())
        )
    except Exception as exc:
        logger.warning("Failed to initialize external MCP servers: %s", exc)


def external_mcp_status_op() -> Dict[str, Any]:
    mgr = get_global_manager()
    if mgr is None:
        return {"enabled": False, "detail": "No external MCP servers configured"}
    status = mgr.status()
    status["enabled"] = True
    return status


def refresh_external_mcp_op() -> Dict[str, Any]:
    mgr = get_global_manager()
    if mgr is None:
        raise RuntimeError("External MCP manager not initialized")
    asyncio.run(mgr.refresh(mcp, tool_decorator=_safe_tool_decorator))
    return external_mcp_status_op()


class MCPBearerAuthMiddleware(BaseHTTPMiddleware):
    """
    Lightweight bearer-token gate for MCP transports. Uses the configured MCP server
    token when require_token is enabled. Local bypass respects the existing auth flag.
    """

    def __init__(self, app, *, token: Optional[str], require_token: bool, allow_local_bypass: bool):
        super().__init__(app)
        self.token = token
        self.require_token = require_token
        self.allow_local_bypass = allow_local_bypass

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else None

        # Local bypass if explicitly allowed
        if self.allow_local_bypass and is_local_connection(client_ip):
            return await call_next(request)

        if not self.require_token:
            return await call_next(request)

        token = None
        auth_header = request.headers.get("authorization")
        api_key_header = request.headers.get("x-api-key")

        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()
        if token is None and api_key_header:
            token = api_key_header.strip()

        if self.token and token == self.token:
            return await call_next(request)

        return JSONResponse(
            {"error": "unauthorized", "detail": "Valid MCP bearer token required"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )





def check_authentication(
    request_headers: Optional[Dict[str, str]] = None,
    client_ip: Optional[str] = None
) -> bool:
    """
    Check if request is authenticated.
    
    Args:
        request_headers: HTTP request headers (if available)
        client_ip: Client IP address
    
    Returns:
        True if authenticated or auth disabled, False otherwise
    """

    return _security_check_authentication(
        request_headers=request_headers,
        client_ip=client_ip,
        settings=_get_auth_settings_override(),
    )


def check_ip_whitelist(client_ip: Optional[str] = None) -> bool:
    """
    Check if client IP is whitelisted.
    
    Args:
        client_ip: Client IP address
    
    Returns:
        True if IP is allowed or whitelist is empty, False otherwise
    """

    return _security_check_ip_whitelist(
        client_ip,
        settings=_get_auth_settings_override(),
    )

# --------------------------------------------------------------------
# Repo configuration
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Repository Configuration
# --------------------------------------------------------------------

# Load repositories from config
# Backwards-compatible: build simple REPOS mapping name->Path and
# REPO_OPTIONS mapping name->{path, respect_gitignore}
REPO_OPTIONS: Dict[str, dict] = {}
REPOS: Dict[str, Path] = {}
for name, info in config.repositories_config.items():
    try:
        p = Path(info.get("path")).expanduser().resolve()
    except Exception:
        continue
    # Persist per-repo runtime options including ignore_patterns
    REPO_OPTIONS[name] = {
        "path": p,
        "respect_gitignore": bool(info.get("respect_gitignore", True)),
        "ignore_patterns": list(info.get("ignore_patterns", []) or []),
    }
    REPOS[name] = p

if not REPOS:
    logger.warning(
        "No repositories configured; starting MCP server with NO repositories."
    )
else:
    logger.info(
        "Configured %d repos: %s",
        len(REPOS),
        ", ".join(f"{k}={v}" for k, v in REPOS.items()),
    )


def _get_repo_root(name: str) -> Path:
    """Lookup a repo root by name."""
    try:
        root = REPOS[name]
        # Ensure we return a Path object
        if isinstance(root, str):
            return Path(root)
        return root
    except KeyError:
        # Fallback: try to read from the index database if present
        try:
            index = _get_index()
            cursor = index.repos_db.execute("SELECT path FROM repos WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return Path(row[0])
        except Exception:
            # Ignore and raise the original error below
            pass

        raise ValueError(f"Unknown repo {name!r}. Known repos: {sorted(REPOS.keys())}")


def _resolve_under_repo(repo: str, rel_path: str) -> Path:
    """
    Resolve a relative path safely under the named repo.

    Raises ValueError if the path escapes the repo root.
    """
    root = _get_repo_root(repo)
    candidate = (root / rel_path).resolve()

    # Ensure candidate is under root (prevent directory traversal)
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Resolved path {candidate} escapes repo root {root} "
            f"(rel_path={rel_path!r})"
        )

    return candidate


def _ensure_repos_configured() -> None:
    if not REPOS:
        raise RuntimeError(
            "No repositories are configured. "
            "Configure repositories in config.json or SIGIL_REPO_MAP."
        )


# --------------------------------------------------------------------
# Index instance (lazy initialization)
# --------------------------------------------------------------------

_INDEX: Optional[SigilIndex] = None
_WATCHER: Optional[FileWatchManager] = None


# --------------------------------------------------------------------
# Shared operational helpers (index and vector index)
# --------------------------------------------------------------------

def rebuild_index_op(
    repo: Optional[str] = None,
    force_rebuild: bool = False,
) -> Dict[str, object]:
    """
    Rebuild the trigram/symbol index for one or all repositories.
    Uses the same logic as scripts/rebuild_indexes.py script.
    Used by both MCP tools and the admin API.
    """
    _ensure_repos_configured()
    index = _get_index()
    
    # Wait a moment for any active file watcher operations to complete
    # This helps prevent database lock conflicts
    import time
    time.sleep(0.5)

    if repo is not None:
        # Per-repo rebuild: use script's single-repo logic
        from sigil_mcp.scripts.rebuild_indexes import rebuild_single_repo_index
        repo_path = _get_repo_root(repo)
        return rebuild_single_repo_index(
            index=index,
            repo_name=repo,
            repo_path=repo_path,
            rebuild_embeddings=False,  # Only rebuild trigrams/symbols
        )

    # Complete rebuild: use script's rebuild_all_indexes logic
    from sigil_mcp.scripts.rebuild_indexes import rebuild_all_indexes
    return rebuild_all_indexes(
        index=index,
        wipe_index=False,  # Don't wipe - we're using existing index
        rebuild_embeddings=False,  # Only rebuild trigrams/symbols
    )


def build_vector_index_op(
    repo: Optional[str] = None,
    force_rebuild: bool = False,
    model: str = "default",
) -> Dict[str, object]:
    """
    Build or refresh the vector index for one or all repositories.
    Uses the same logic as scripts/rebuild_indexes.py script.
    """
    cfg = get_config()
    _ensure_repos_configured()
    index = _get_index()
    if not cfg.embeddings_enabled:
        logger.info("Vector rebuild requested but embeddings are disabled.")
        return {
            "success": False,
            "status": "skipped",
            "reason": "embeddings_disabled",
            "message": "Embeddings are disabled; trigram search only.",
        }
    if not getattr(index, "lancedb_available", True):
        logger.warning(
            "Vector rebuild requested but LanceDB is unavailable. "
            "Install optional dependency group 'lancedb'."
        )
        return {
            "success": False,
            "status": "skipped",
            "reason": "lancedb_missing",
            "message": "Install lancedb optional dependencies to enable vector indexing.",
        }
    
    # Wait a moment for any active file watcher operations to complete
    # This helps prevent database lock conflicts
    import time
    time.sleep(0.5)

    # Ensure embeddings are configured
    if index.embed_fn is None:
        from .embeddings import create_embedding_provider
        import numpy as np
        
        if not cfg.embeddings_enabled:
            raise RuntimeError("Embeddings not enabled in configuration")
        
        provider = cfg.embeddings_provider
        model_name = cfg.embeddings_model or model
        
        if not provider or not model_name:
            raise RuntimeError("Embedding provider/model not configured")
        
        kwargs = dict(cfg.embeddings_kwargs)
        if cfg.embeddings_cache_dir:
            kwargs["cache_dir"] = cfg.embeddings_cache_dir
        if provider == "openai" and cfg.embeddings_api_key:
            kwargs["api_key"] = cfg.embeddings_api_key
        
        try:
            embedding_provider = create_embedding_provider(
                provider=provider,
                model=model_name,
                dimension=cfg.embeddings_dimension,
                **kwargs
            )
        except (ImportError, FileNotFoundError) as exc:
            logger.error(
                "Cannot rebuild vector index - embedding backend unavailable: %s",
                exc,
            )
            return {
                "success": False,
                "status": "skipped",
                "reason": "embeddings_unavailable",
                "message": str(exc),
            }
        
        def embed_fn(texts):
            embeddings_list = embedding_provider.embed_documents(list(texts))
            return np.array(embeddings_list, dtype="float32")
        
        index.embed_fn = embed_fn
        index.embed_model = f"{provider}:{model_name}"

    if repo is not None:
        repo_path = _get_repo_root(repo)
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

        repo_stats = index.build_vector_index(
            repo=repo,
            embed_fn=index.embed_fn,
            model=index.embed_model,
            force=force_rebuild,
        )
        return {
            "success": True,
            "status": "completed",
            "repo": repo,
            "model": index.embed_model,
            "message": f"Successfully rebuilt vector index for {repo}",
            "stats": {
                "documents": repo_stats.get("documents_processed", 0),
                "symbols": 0,  # Vector index doesn't track symbols
                "files": repo_stats.get("documents_processed", 0),
            },
            **repo_stats,
        }

    embedding_stats: Dict[str, Dict[str, object]] = {}
    total_docs = 0
    for repo_name in REPOS.keys():
        try:
            stats = index.build_vector_index(
                repo=repo_name,
                embed_fn=index.embed_fn,
                model=index.embed_model,
                force=force_rebuild,
            )
        except Exception:
            logger.exception("Failed to rebuild vector index for %s", repo_name)
            raise

        embedding_stats[repo_name] = {k: cast(object, v) for k, v in stats.items()}
        total_docs += stats.get("documents_processed", 0)

    return {
        "success": True,
        "status": "completed",
        "model": index.embed_model,
        "message": (
            f"Successfully rebuilt vector index for "
            f"{len(embedding_stats)} repositories"
        ),
        "stats": {
            "documents": total_docs,
            "symbols": 0,  # Vector index doesn't track symbols
            "files": total_docs,
        },
        "repos": embedding_stats,
    }


def get_index_stats_op(repo: Optional[str] = None) -> Dict[str, object]:
    """
    Thin wrapper around SigilIndex.get_index_stats.
    Returns stats in format expected by Admin UI frontend.
    """
    _ensure_repos_configured()
    index = _get_index()

    vector_count_failed = False

    def _count_vectors(filter_expr: Optional[str] = None) -> int:
        """Count vectors without materializing the table."""
        if index.vectors is None:
            return 0

        if hasattr(index.vectors, "count_rows"):
            return int(index.vectors.count_rows(filter=filter_expr))

        # Fallback: best-effort zero when count_rows API missing
        return 0

    def _get_vector_count(repo_name: Optional[str] = None) -> int:
        """Count vectors stored in LanceDB, optionally filtered by repo."""
        nonlocal vector_count_failed

        def _mark_stale_on_failure() -> None:
            try:
                if hasattr(index, "_vector_index_stale"):
                    index._vector_index_stale = True
            except Exception:
                logger.debug(
                    "Unable to mark vector index stale after count failure",
                    exc_info=True,
                )

        if index.vectors is None:
            return 0

        if repo_name is None:
            try:
                return _count_vectors()
            except Exception as exc:
                if not vector_count_failed:
                    logger.exception(
                        "Failed to count vectors across all repositories; "
                        "marking vector index stale (rebuild may be required)"
                    )
                    vector_count_failed = True
                else:
                    logger.error(
                        "Failed to count vectors across all repositories: %s",
                        exc,
                    )
                _mark_stale_on_failure()
                return 0

        cursor = index.repos_db.cursor()
        cursor.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
        row = cursor.fetchone()
        if not row:
            return 0

        repo_id = str(row[0])

        try:
            return _count_vectors(f"repo_id == '{repo_id}'")
        except Exception as exc:
            if not vector_count_failed:
                logger.exception(
                    "Failed to count vectors for repo %s; marking vector index stale "
                    "(consider rebuild_embeddings or removing LanceDB directory)",
                    repo_name,
                )
                vector_count_failed = True
            else:
                logger.error("Failed to count vectors for repo %s: %s", repo_name, exc)
            _mark_stale_on_failure()
            return 0

    stats = index.get_index_stats(repo=repo)
    
    # Handle error response
    if isinstance(stats, dict) and "error" in stats:
        return dict(stats)  # type: ignore[return-value]
    
    # Transform to match frontend expectations
    if isinstance(stats, dict):
        # If repo-specific, return as-is but ensure structure
        if repo:
            # Get file count for this repo
            from .server import REPOS
            repo_path = Path(REPOS[repo])
            file_count = (
                sum(1 for _ in repo_path.rglob("*") if _.is_file())
                if repo_path.exists()
                else 0
            )
            
            # compute stale vectors count for this repo from repos.db
            try:
                cur = index.repos_db.cursor()
                cur.execute("SELECT id FROM repos WHERE name = ?", (repo,))
                r = cur.fetchone()
                if r:
                    repo_id = r[0]
                    cur.execute(
                        "SELECT COUNT(*) FROM documents WHERE repo_id = ? AND (vector_index_error IS NOT NULL OR vector_indexed_at IS NULL)",
                        (repo_id,),
                    )
                    vectors_stale = int(cur.fetchone()[0] or 0)
                else:
                    vectors_stale = 0
            except Exception:
                logger.debug("Failed to compute vectors_stale for repo %s", repo, exc_info=True)
                vectors_stale = 0

            return {
                "total_documents": stats.get("documents", 0),
                "total_symbols": stats.get("symbols", 0),
                "total_vectors": _get_vector_count(repo),
                "total_vectors_stale": vectors_stale,
                "total_repos": 1,
                "repos": {
                    repo: {
                        "documents": stats.get("documents", 0),
                        "symbols": stats.get("symbols", 0),
                        "files": file_count,
                        "vectors": _get_vector_count(repo),
                        "vectors_stale": vectors_stale,
                    }
                }
            }
        # Aggregate stats for all repos - need to get per-repo breakdown
        total_docs = stats.get("documents", 0)
        total_symbols = stats.get("symbols", 0)
        total_repos = stats.get("repositories", 0)
        total_vectors = _get_vector_count()

        # compute total stale vectors across all repos
        try:
            cur = index.repos_db.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM documents WHERE (vector_index_error IS NOT NULL OR vector_indexed_at IS NULL)"
            )
            total_vectors_stale = int(cur.fetchone()[0] or 0)
        except Exception:
            logger.debug("Failed to compute total_vectors_stale", exc_info=True)
            total_vectors_stale = 0

        # Get per-repo stats and per-repo stale counts
        repos_dict: Dict[str, Dict[str, int]] = {}
        from .server import REPOS
        for repo_name in REPOS.keys():
            repo_stats = index.get_index_stats(repo=repo_name)
            if isinstance(repo_stats, dict) and "error" not in repo_stats:
                repo_path = Path(REPOS[repo_name])
                file_count = (
                    sum(1 for _ in repo_path.rglob("*") if _.is_file())
                    if repo_path.exists()
                    else 0
                )
                # compute per-repo stale count
                try:
                    cur = index.repos_db.cursor()
                    cur.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
                    row = cur.fetchone()
                    if row:
                        repo_id = row[0]
                        cur.execute(
                            "SELECT COUNT(*) FROM documents WHERE repo_id = ? AND (vector_index_error IS NOT NULL OR vector_indexed_at IS NULL)",
                            (repo_id,),
                        )
                        repo_vectors_stale = int(cur.fetchone()[0] or 0)
                    else:
                        repo_vectors_stale = 0
                except Exception:
                    logger.debug("Failed to compute vectors_stale for %s", repo_name, exc_info=True)
                    repo_vectors_stale = 0

                repos_dict[repo_name] = {
                    "documents": int(repo_stats.get("documents", 0)),
                    "symbols": int(repo_stats.get("symbols", 0)),
                    "files": int(file_count),
                    "vectors": _get_vector_count(repo_name),
                    "vectors_stale": repo_vectors_stale,
                }

        return {
            "total_documents": total_docs,
            "total_symbols": total_symbols,
            "total_repos": total_repos,
            "total_vectors": total_vectors,
            "total_vectors_stale": total_vectors_stale,
            "repos": repos_dict
        }
    
    return {"error": "invalid_response", "detail": "Unexpected stats format"}


def _create_embedding_function():
    """
    Create embedding function based on configuration.
    
    Returns:
        Tuple of (embed_fn, model_name) or (None, None) if embeddings disabled
    """
    global READINESS

    if not config.embeddings_enabled:
        logger.info("Embeddings disabled in config")
        READINESS["embeddings"] = True
        return None, None
    
    provider = config.embeddings_provider
    if not provider:
        logger.warning(
            "Embeddings enabled but no provider configured. "
            "Set embeddings.provider in config.json. Embeddings disabled."
        )
        return None, None
    
    try:
        from sigil_mcp.embeddings import create_embedding_provider
        
        model = config.embeddings_model
        if not model:
            logger.error(
                f"Embeddings provider '{provider}' requires model configuration. "
                "Set embeddings.model in config.json. Embeddings disabled."
            )
            return None, None
        
        dimension = config.embeddings_dimension
        
        # Build kwargs for provider
        kwargs = dict(config.embeddings_kwargs)
        if config.embeddings_cache_dir:
            kwargs["cache_dir"] = config.embeddings_cache_dir
        if provider == "openai" and config.embeddings_api_key:
            kwargs["api_key"] = config.embeddings_api_key
        
        logger.info(f"Initializing {provider} embedding provider with model: {model}")
        embedding_provider = create_embedding_provider(
            provider=provider,
            model=model,
            dimension=dimension,
            **kwargs
        )
        
        # Create wrapper function that matches SigilIndex expectations
        def embed_fn(texts: Sequence[str]) -> np.ndarray:
            embeddings_list = embedding_provider.embed_documents(list(texts))
            return np.array(embeddings_list, dtype="float32")
        
        model_name = f"{provider}:{model}"
        logger.info(f"Embeddings initialized: {model_name} (dim={dimension})")
        READINESS["embeddings"] = True
        return embed_fn, model_name
        
    except ImportError as e:
        logger.error(
            f"Failed to import embedding provider '{provider}': {e}. "
            "Install required dependencies. See docs/EMBEDDING_SETUP.md. "
            "Embeddings disabled."
        )
        READINESS["embeddings"] = False
        return None, None
    except FileNotFoundError as e:
        logger.error(
            "Embedding model not found: %s. "
            "Download the configured model (e.g., Jina GGUF) into ./models "
            "or set embeddings.model to the correct path. Falling back to trigram search.",
            e,
        )
        READINESS["embeddings"] = False
        return None, None
    except Exception as e:
        logger.error(
            f"Failed to initialize embedding provider '{provider}': {e}. "
            "Embeddings disabled."
        )
        READINESS["embeddings"] = False
        return None, None


def _get_index() -> SigilIndex:
    """Get or create the global index instance."""
    global _INDEX
    if _INDEX is None:
        index_path = config.index_path
        embed_fn, embed_model = _create_embedding_function()
        # Provide default embed_model if None
        _INDEX = SigilIndex(
            index_path,
            embed_fn=embed_fn,
            embed_model=embed_model if embed_model else "none"
        )
        READINESS["index"] = True
        logger.info(f"Initialized index at {index_path}")
    return _INDEX


def _attempt_local_index_remove(repo_name: str, repo_path: Path, file_path: Path) -> bool:
    """Attempt to find a nearby index directory and remove a file using it.

    This scans one level up from `repo_path` and looks for folders containing
    `repos.db`. If found, it instantiates a temporary `SigilIndex` and attempts
    to remove the file. Returns True if removed using a local index.
    """
    parent = repo_path.parent
    logger.warning(
        "_on_file_change: attempting local fallback under parent %s",
        parent,
    )
    tried_local = False
    for entry in parent.iterdir():
        if not entry.is_dir():
            continue
        db = entry / "repos.db"
        if not db.exists():
            continue
        tried_local = True
        logger.warning(
            "_on_file_change: found local index at %s, trying removal",
            entry,
        )
        try:
            local_index = SigilIndex(entry)
            try:
                removed = local_index.remove_file(repo_name, repo_path, file_path)
            finally:
                try:
                    local_index.repos_db.close()
                except Exception:
                    pass
                try:
                    if getattr(local_index, "trigrams_db", None):
                        local_index.trigrams_db.close()
                except Exception:
                    pass
            if removed:
                logger.info("Removed %s from local index at %s", file_path, entry)
                return True
            else:
                logger.debug("Local index at %s did not contain file %s", entry, file_path)
        except Exception:
            logger.exception("Failed to handle local index at %s", entry)

    if not tried_local:
        logger.debug("_on_file_change: no nearby index dir with repos.db under %s", parent)
    return False


def _handle_deleted_event(
    index: SigilIndex, repo_name: str, repo_path: Path, file_path: Path
) -> bool:
    """Handle a filesystem 'deleted' event by trying global index then local fallbacks.

    Returns True if the file was removed from an index.
    """
    removed = False
    try:
        removed = index.remove_file(repo_name, repo_path, file_path)
        logger.warning("_on_file_change: removed using global index -> %s", removed)
    except ValueError:
        logger.debug("_on_file_change: global index raised ValueError; repo unknown")
    if not removed:
        try:
            return _attempt_local_index_remove(repo_name, repo_path, file_path)
        except Exception:
            logger.exception("Error attempting local index fallback for %s", repo_name)
            return False
    return removed


def _on_file_change(repo_name: str, file_path: Path, event_type: str):
    """Handle file change events from watcher."""
    logger.info(f"File {event_type}: {file_path.name} in {repo_name}")

    try:
        index = _get_index()
        repo_path = _get_repo_root(repo_name)

        if event_type == "deleted":
            _handle_deleted_event(index, repo_name, repo_path, file_path)
            return

        # Ensure embedding function is initialized on-demand when possible
        try:
            if getattr(index, "embed_fn", None) is None and get_config().embeddings_enabled:
                try:
                    embed_fn, embed_model = _create_embedding_function()
                    if embed_fn:
                        index.embed_fn = embed_fn
                        index.embed_model = embed_model or index.embed_model
                        logger.info("Initialized embed_fn on-demand in _on_file_change")
                except Exception:
                    logger.debug("On-demand embedding init failed; continuing without embeddings", exc_info=True)
        except Exception:
            logger.debug("Failed to check/init embed_fn in _on_file_change", exc_info=True)

        # Granular re-indexing for modified/created files — run in background to avoid blocking watcher
        def _do_index():
            try:
                success = index.index_file(repo_name, repo_path, file_path)
                if success:
                    logger.info(f"Re-indexed {file_path.name} after {event_type}")
                else:
                    logger.debug(f"Skipped re-indexing {file_path.name}")
            except Exception as e:
                logger.error(f"Error during background re-index of {file_path}: {e}")

        t = threading.Thread(target=_do_index, daemon=True)
        t.start()
    except Exception as e:
        logger.error(f"Error re-indexing after {event_type}: {e}")


def _get_watcher() -> Optional[FileWatchManager]:
    """Get or create the global file watcher."""
    global _WATCHER
    
    if not config.watch_enabled:
        READINESS["watcher"] = True
        return None
    
    if _WATCHER is None:
        _WATCHER = FileWatchManager(
            on_change=_on_file_change,
            ignore_dirs=config.watch_ignore_dirs,
            ignore_extensions=config.watch_ignore_extensions,
        )
        _WATCHER.start()
        logger.info("File watcher initialized")
        READINESS["watcher"] = bool(getattr(_WATCHER, "enabled", True))
    else:
        READINESS["watcher"] = bool(_WATCHER and getattr(_WATCHER, "enabled", True))
    
    return _WATCHER


def _start_watching_repos():
    """Start watching all configured repositories."""
    watcher = _get_watcher()
    if watcher:
        for repo_name, repo_path in REPOS.items():
            # honor per-repo respect_gitignore option when creating the watcher
            respect = True
            try:
                respect = bool(REPO_OPTIONS.get(repo_name, {}).get("respect_gitignore", True))
            except Exception:
                respect = True
            watcher.watch_repository(repo_name, Path(repo_path), honor_gitignore=respect)


# --------------------------------------------------------------------
# OAuth HTTP Endpoints (Standard OAuth 2.0 Protocol)
# --------------------------------------------------------------------


@mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
async def oauth_metadata(request: Request) -> JSONResponse:
    """OAuth 2.0 Authorization Server Metadata (RFC 8414)."""
    if not OAUTH_ENABLED:
        return JSONResponse({"error": "OAuth not enabled"}, status_code=501)
    
    base_url = str(request.base_url).rstrip('/')
    
    response = JSONResponse({
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/oauth/authorize",
        "token_endpoint": f"{base_url}/oauth/token",
        "revocation_endpoint": f"{base_url}/oauth/revoke",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": [
            "client_secret_post", "client_secret_basic", "none"
        ],
        "code_challenge_methods_supported": ["S256", "plain"]
    })
    
    # Add ngrok bypass header
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


@mcp.custom_route("/.well-known/openid-configuration", methods=["GET"])
async def openid_configuration(request: Request) -> JSONResponse:
    """OpenID Connect Discovery (for ChatGPT compatibility)."""
    if not OAUTH_ENABLED:
        return JSONResponse({"error": "OAuth not enabled"}, status_code=501)
    
    base_url = str(request.base_url).rstrip('/')
    
    response = JSONResponse({
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/oauth/authorize",
        "token_endpoint": f"{base_url}/oauth/token",
        "revocation_endpoint": f"{base_url}/oauth/revoke",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": [
            "client_secret_post", "client_secret_basic", "none"
        ],
        "code_challenge_methods_supported": ["S256", "plain"]
    })
    
    # Add ngrok bypass header
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


@mcp.custom_route("/healthz", methods=["GET"])  # lightweight HTTP healthcheck
async def healthz(request: Request) -> JSONResponse:
    """Lightweight health endpoint used by service monitors and cloudflared healthchecks.

    Returns 200 with a simple JSON payload so that external monitoring can verify the
    tunnel and backend availability.
    """
    return JSONResponse({"status": "ok"}, status_code=200)


def _is_ready() -> Dict[str, bool]:
    """Return readiness flags and attempt lazy initialization if needed."""

    if not READINESS.get("index"):
        try:
            _get_index()
        except Exception as exc:  # pragma: no cover - logged for operators
            logger.exception("Readiness check failed to initialize index: %s", exc)
    return dict(READINESS)


@mcp.custom_route("/readyz", methods=["GET"])
async def readyz(request: Request) -> JSONResponse:
    """Readiness endpoint for orchestration systems (returns 503 until ready)."""

    readiness = _is_ready()
    ready = all(readiness.values())
    status_code = 200 if ready else 503
    return JSONResponse({"ready": ready, "components": readiness}, status_code=status_code)


@mcp.custom_route("/oauth/authorize", methods=["GET", "POST"])
async def oauth_authorize_http(
    request: Request
) -> JSONResponse | RedirectResponse | HTMLResponse:
    """OAuth 2.0 Authorization Endpoint."""
    logger.info("="*80)
    logger.info(f"OAuth authorization request received - Method: {request.method}")
    logger.info(f"Request URL: {request.url}")
    # Headers are now logged by HeaderLoggingASGIMiddleware (redacted)
    
    if not OAUTH_ENABLED:
        return JSONResponse({"error": "oauth_not_enabled"}, status_code=501)
    
    # Get parameters from query string or form data
    if request.method == "GET":
        params = dict(request.query_params)
    else:
        form = await request.form()
        # Convert form values to strings (form() returns str | UploadFile)
        params = {k: get_form_value(v) for k, v in form.items()}
    
    client_id = params.get("client_id")
    redirect_uri = params.get("redirect_uri")
    response_type = params.get("response_type", "code")
    state = params.get("state")
    scope = params.get("scope")
    code_challenge = params.get("code_challenge")
    code_challenge_method = params.get("code_challenge_method")
    
    # Validate required parameters
    if not client_id or not redirect_uri:
        return JSONResponse({
            "error": "invalid_request",
            "error_description": "client_id and redirect_uri are required"
        }, status_code=400)
    
    if response_type != "code":
        return JSONResponse({
            "error": "unsupported_response_type",
            "error_description": "Only 'code' response type is supported"
        }, status_code=400)
    
    # Verify client
    oauth_manager = get_oauth_manager()
    if not oauth_manager.verify_client(client_id):
        return JSONResponse({
            "error": "invalid_client",
            "error_description": "Invalid client_id"
        }, status_code=401)
    
    # Verify redirect_uri
    client = oauth_manager.get_client()
    if not client:
        return JSONResponse({
            "error": "server_error",
            "error_description": "OAuth client not configured"
        }, status_code=500)
    
    if not _security_is_redirect_uri_allowed(
        redirect_uri,
        client.redirect_uris,
        config.oauth_redirect_allow_list,
    ):
        return JSONResponse({
            "error": "invalid_request",
            "error_description": "Redirect URI must be registered or in allow list"
        }, status_code=400)
    
    # Check if this is a consent approval (POST with approve=true)
    logger.info(f"All params: {params}")
    if request.method == "POST" and params.get("approve") == "true":
        logger.info("User approved consent - generating authorization code")
        # User approved - generate code and redirect
        code = oauth_manager.create_authorization_code(
            client_id=client_id,
            redirect_uri=redirect_uri,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method
        )
        
        # Build redirect URL
        redirect_params = {"code": code}
        if state:
            redirect_params["state"] = state
        
        redirect_url = f"{redirect_uri}?{urlencode(redirect_params)}"
        
        logger.info(f"Redirecting to: {redirect_url}")
        logger.info(f"Authorization code: {code[:20]}...")
        logger.info("="*80)
        return RedirectResponse(redirect_url, status_code=302)
    
    # Show consent screen (GET request or initial POST)
    
    # Build approval form
    consent_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Authorize Sigil MCP Server</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .container {{
                background: white;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                max-width: 500px;
                width: 90%;
            }}
            h1 {{
                color: #333;
                margin: 0 0 1rem 0;
                font-size: 1.5rem;
            }}
            .info {{
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                border-left: 4px solid #667eea;
            }}
            .info p {{
                margin: 0.5rem 0;
                color: #666;
                font-size: 0.9rem;
            }}
            .info strong {{
                color: #333;
            }}
            .buttons {{
                display: flex;
                gap: 1rem;
                margin-top: 1.5rem;
            }}
            button {{
                flex: 1;
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 6px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .approve {{
                background: #667eea;
                color: white;
            }}
            .approve:hover {{
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }}
            .deny {{
                background: #e0e0e0;
                color: #666;
            }}
            .deny:hover {{
                background: #d0d0d0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>馃攼 Authorize Access</h1>
            <p><strong>ChatGPT</strong> is requesting access to your Sigil MCP Server.</p>
            
            <div class="info">
                <p><strong>Client:</strong> {client_id[:20]}...</p>
                <p><strong>Scope:</strong> {scope or "Default access"}</p>
                <p><strong>This will allow:</strong></p>
                <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                    <li>Reading repository information</li>
                    <li>Searching code and files</li>
                    <li>Accessing configured tools</li>
                </ul>
            </div>
            
            <form method="POST" action="/oauth/authorize">
                <input type="hidden" name="client_id" value="{client_id}">
                <input type="hidden" name="redirect_uri" value="{redirect_uri}">
                <input type="hidden" name="response_type" value="{response_type}">
                <input type="hidden" name="state" value="{state or ''}">
                <input type="hidden" name="scope" value="{scope or ''}">
                <input type="hidden" name="code_challenge" value="{code_challenge or ''}">
                <input type="hidden" name="code_challenge_method" 
                       value="{code_challenge_method or ''}">
                <input type="hidden" name="approve" value="true">
                
                <div class="buttons">
                    <button type="submit" class="approve">Authorize</button>
                    <button type="button" class="deny" onclick="window.close()">Deny</button>
                </div>
            </form>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=consent_html, status_code=200)


@mcp.custom_route("/oauth/token", methods=["POST"])
async def oauth_token_http(request: Request) -> JSONResponse:
    """OAuth 2.0 Token Endpoint."""
    logger.info("="*80)
    logger.info("OAuth token request received")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request URL: {request.url}")
    # Headers are now logged by HeaderLoggingASGIMiddleware (redacted)
    
    if not OAUTH_ENABLED:
        return JSONResponse({"error": "oauth_not_enabled"}, status_code=501)
    
    # Parse form data
    form = await request.form()
    grant_type = get_form_value(form.get("grant_type"))
    code = get_form_value(form.get("code"))
    redirect_uri = get_form_value(form.get("redirect_uri"))
    client_id = get_form_value(form.get("client_id"))
    client_secret = get_form_value(form.get("client_secret"))
    code_verifier = get_form_value(form.get("code_verifier"))
    refresh_token = get_form_value(form.get("refresh_token"))
    
    logger.info(
        f"Form data: grant_type={grant_type}, code={code[:20] if code else None}..., "
        f"redirect_uri={redirect_uri}, client_id={client_id}, "
        f"client_secret={'***' if client_secret else None}, "
        f"code_verifier={code_verifier[:20] if code_verifier else None}..., "
        f"refresh_token={refresh_token[:20] if refresh_token else None}..."
    )
    
    # Check for client credentials in Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Basic "):
        import base64
        try:
            decoded = base64.b64decode(auth_header[6:]).decode()
            header_client_id, header_client_secret = decoded.split(":", 1)
            client_id = client_id or header_client_id
            client_secret = client_secret or header_client_secret
        except Exception:
            pass
    
    if not client_id:
        return JSONResponse({
            "error": "invalid_request",
            "error_description": "client_id is required"
        }, status_code=400)
    
    oauth_manager = get_oauth_manager()
    
    # Verify client (public clients don't need secret)
    if not oauth_manager.verify_client(client_id, client_secret):
        return JSONResponse({
            "error": "invalid_client",
            "error_description": "Invalid client credentials"
        }, status_code=401)
    
    if grant_type == "authorization_code":
        if not code or not redirect_uri:
            return JSONResponse({
                "error": "invalid_request",
                "error_description": "code and redirect_uri are required"
            }, status_code=400)
        
        token = oauth_manager.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier
        )
        
        if not token:
            return JSONResponse({
                "error": "invalid_grant",
                "error_description": "Invalid authorization code"
            }, status_code=400)
        
        return JSONResponse({
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expires_in": token.expires_in,
            "refresh_token": token.refresh_token,
            "scope": token.scope
        })
    
    elif grant_type == "refresh_token":
        if not refresh_token:
            return JSONResponse({
                "error": "invalid_request",
                "error_description": "refresh_token is required"
            }, status_code=400)
        
        token = oauth_manager.refresh_access_token(refresh_token)
        
        if not token:
            return JSONResponse({
                "error": "invalid_grant",
                "error_description": "Invalid refresh token"
            }, status_code=400)
        
        return JSONResponse({
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expires_in": token.expires_in,
            "refresh_token": token.refresh_token,
            "scope": token.scope
        })
    
    else:
        return JSONResponse({
            "error": "unsupported_grant_type",
            "error_description": f"Grant type '{grant_type}' is not supported"
        }, status_code=400)


@mcp.custom_route("/oauth/revoke", methods=["POST"])
async def oauth_revoke_http(request: Request) -> JSONResponse:
    """OAuth 2.0 Token Revocation Endpoint (RFC 7009)."""
    logger.info("OAuth revocation request received")
    
    if not OAUTH_ENABLED:
        return JSONResponse({"error": "oauth_not_enabled"}, status_code=501)
    
    form = await request.form()
    token = get_form_value(form.get("token"))
    client_id = get_form_value(form.get("client_id"))
    client_secret = get_form_value(form.get("client_secret"))
    
    if not token or not client_id:
        return JSONResponse({
            "error": "invalid_request",
            "error_description": "token and client_id are required"
        }, status_code=400)
    
    oauth_manager = get_oauth_manager()
    
    # Verify client
    if not oauth_manager.verify_client(client_id, client_secret):
        return JSONResponse({
            "error": "invalid_client",
            "error_description": "Invalid client credentials"
        }, status_code=401)
    
    oauth_manager.revoke_token(token)
    
    # RFC 7009: The revocation endpoint returns 200 even if token doesn't exist
    return JSONResponse({"status": "revoked"})


# --------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# OAuth Endpoints (MCP Tools - for manual testing)
# --------------------------------------------------------------------


def oauth_authorize(
    client_id: str,
    redirect_uri: str,
    response_type: str = "code",
    state: Optional[str] = None,
    scope: Optional[str] = None,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None
) -> Dict[str, str]:
    """
    OAuth2 authorization endpoint.
    
    Initiates the OAuth2 authorization code flow. This should be called
    by the OAuth client (e.g., ChatGPT) to request authorization.
    
    Args:
        client_id: OAuth client ID
        redirect_uri: URI to redirect to after authorization
        response_type: Must be "code" for authorization code flow
        state: CSRF protection token (recommended)
        scope: Requested permissions (optional)
        code_challenge: PKCE code challenge (recommended)
        code_challenge_method: PKCE method, "S256" or "plain"
    
    Returns:
        Authorization response with redirect URL or error
    
    Example:
        oauth_authorize(
            client_id="sigil_xxx",
            redirect_uri="http://localhost:8080/oauth/callback",
            state="random_state_value",
            code_challenge="challenge_hash"
        )
    """
    logger.info(
        "oauth_authorize called (client_id=%r, redirect_uri=%r)",
        client_id,
        redirect_uri
    )
    
    if not OAUTH_ENABLED:
        return {
            "error": "oauth_not_enabled",
            "error_description": "OAuth is disabled on this server"
        }
    
    # Verify response_type
    if response_type != "code":
        return {
            "error": "unsupported_response_type",
            "error_description": "Only 'code' response type is supported"
        }
    
    # Verify client
    oauth_manager = get_oauth_manager()
    if not oauth_manager.verify_client(client_id):
        return {
            "error": "invalid_client",
            "error_description": "Invalid client_id"
        }
    
    # Verify redirect_uri
    client = oauth_manager.get_client()
    allowed_redirects = config.oauth_redirect_allow_list
    if not client or not _security_is_redirect_uri_allowed(
        redirect_uri or "",
        client.redirect_uris,
        allowed_redirects,
    ):
        return {
            "error": "invalid_request",
            "error_description": "Redirect URI not registered or allowed for this client"
        }
    
    # Auto-approve (for trusted clients)
    # In production, you might want a consent screen here
    code = oauth_manager.create_authorization_code(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method
    )
    
    # Build redirect URL
    params = {"code": code}
    if state:
        params["state"] = state
    
    redirect_url = f"{redirect_uri}?{urlencode(params)}"
    
    result: Dict[str, str] = {
        "redirect_url": redirect_url,
        "code": code
    }
    if state:
        result["state"] = state
    
    return result


def oauth_token(
    grant_type: str,
    code: Optional[str] = None,
    redirect_uri: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    code_verifier: Optional[str] = None,
    refresh_token: Optional[str] = None
) -> Dict[str, object]:
    """
    OAuth2 token endpoint.
    
    Exchange an authorization code for an access token, or refresh
    an existing access token.
    
    Args:
        grant_type: "authorization_code" or "refresh_token"
        code: Authorization code (for authorization_code grant)
        redirect_uri: Redirect URI (must match authorization request)
        client_id: OAuth client ID
        client_secret: OAuth client secret (for confidential clients)
        code_verifier: PKCE code verifier
        refresh_token: Refresh token (for refresh_token grant)
    
    Returns:
        Token response with access_token, expires_in, etc.
    
    Example:
        # Exchange authorization code
        oauth_token(
            grant_type="authorization_code",
            code="auth_code_here",
            redirect_uri="http://localhost:8080/oauth/callback",
            client_id="sigil_xxx",
            code_verifier="verifier_string"
        )
        
        # Refresh token
        oauth_token(
            grant_type="refresh_token",
            refresh_token="refresh_token_here",
            client_id="sigil_xxx"
        )
    """
    logger.info("oauth_token called (grant_type=%r)", grant_type)
    
    if not OAUTH_ENABLED:
        return {
            "error": "oauth_not_enabled",
            "error_description": "OAuth is disabled on this server"
        }
    
    oauth_manager = get_oauth_manager()
    
    # Validate client_id
    if not client_id:
        return {
            "error": "invalid_request",
            "error_description": "client_id is required"
        }
    
    # Verify client (public clients only need client_id)
    if not oauth_manager.verify_client(client_id, client_secret):
        return {
            "error": "invalid_client",
            "error_description": "Invalid client credentials"
        }
    
    if grant_type == "authorization_code":
        if not code or not redirect_uri:
            return {
                "error": "invalid_request",
                "error_description": "code and redirect_uri are required"
            }
        
        token = oauth_manager.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier
        )
        
        if not token:
            return {
                "error": "invalid_grant",
                "error_description": "Invalid authorization code"
            }
        
        return {
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expires_in": token.expires_in,
            "refresh_token": token.refresh_token,
            "scope": token.scope
        }
    
    elif grant_type == "refresh_token":
        if not refresh_token:
            return {
                "error": "invalid_request",
                "error_description": "refresh_token is required"
            }
        
        token = oauth_manager.refresh_access_token(refresh_token)
        
        if not token:
            return {
                "error": "invalid_grant",
                "error_description": "Invalid refresh token"
            }
        
        return {
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expires_in": token.expires_in,
            "refresh_token": token.refresh_token,
            "scope": token.scope
        }
    
    else:
        return {
            "error": "unsupported_grant_type",
            "error_description": f"Grant type '{grant_type}' is not supported"
        }


def oauth_revoke(
    token: str,
    client_id: str,
    client_secret: Optional[str] = None
) -> Dict[str, str]:
    """
    OAuth2 token revocation endpoint.
    
    Revoke an access token or refresh token, immediately invalidating it.
    
    Args:
        token: Access token or refresh token to revoke
        client_id: OAuth client ID
        client_secret: OAuth client secret (optional)
    
    Returns:
        Status of revocation
    
    Example:
        oauth_revoke(
            token="access_token_here",
            client_id="sigil_xxx"
        )
    """
    logger.info("oauth_revoke called")
    
    if not OAUTH_ENABLED:
        return {
            "error": "oauth_not_enabled",
            "error_description": "OAuth is disabled on this server"
        }
    
    oauth_manager = get_oauth_manager()
    
    # Verify client
    if not oauth_manager.verify_client(client_id, client_secret):
        return {
            "error": "invalid_client",
            "error_description": "Invalid client credentials"
        }
    
    revoked = oauth_manager.revoke_token(token)
    
    if revoked:
        return {"status": "revoked"}
    else:
        return {"status": "not_found"}


def oauth_client_info() -> Dict[str, object]:
    """
    Get OAuth client configuration.
    
    Returns the client_id and allowed redirect_uris for this server.
    Use this to get the credentials needed to configure ChatGPT or
    other OAuth clients.
    
    Returns:
        OAuth client configuration
    
    Example:
        oauth_client_info()
    """
    logger.info("oauth_client_info called")
    
    if not OAUTH_ENABLED:
        return {
            "error": "oauth_not_enabled",
            "error_description": "OAuth is disabled on this server"
        }
    
    oauth_manager = get_oauth_manager()
    client = oauth_manager.get_client()
    
    if not client:
        return {
            "error": "not_configured",
            "error_description": "OAuth client not yet configured. Restart server to initialize."
        }
    
    return {
        "client_id": client.client_id,
        "redirect_uris": client.redirect_uris,
        "authorization_endpoint": "/oauth/authorize",
        "token_endpoint": "/oauth/token",
        "revocation_endpoint": "/oauth/revoke"
    }


# --------------------------------------------------------------------
# MCP Tools
# --------------------------------------------------------------------


@_safe_tool_decorator(
    title=(
        "Read-only 鈥� ping"
    ),
    description=(
        "Healthcheck (read-only). Returns basic server status and configured "
        "repo names."
    ),
    annotations={
        "operation": "read",
        "safety": "non-destructive",
        "audience": ["assistant", "user"],
    },
)
def ping() -> Dict[str, object]:
    """
    Healthcheck endpoint.

    Use this to verify that tools/call is actually happening.
    Returns basic server status and configured repo names.
    """
    import datetime as _dt

    logger.info("ping tool called")

    return {
        "ok": True,
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "repo_count": len(REPOS),
        "repos": sorted(REPOS.keys()),
    }


@_safe_tool_decorator(
    title=(
        "Read-only 鈥� list_repos"
    ),
    description=(
        "List all configured repositories (read-only). Each entry includes "
        "name and absolute path."
    ),
    annotations={
        "operation": "read",
        "safety": "non-destructive",
        "audience": ["assistant", "user"],
    },
)
def list_repos() -> List[Dict[str, str]]:
    """
    List all configured repositories.

    Each entry has:
      - name: Logical repo name
      - path: Absolute filesystem path
    """
    logger.info("list_repos tool called")
    _ensure_repos_configured()

    result = [
        {"name": name, "path": str(path)}
        for name, path in sorted(REPOS.items(), key=lambda kv: kv[0])
    ]
    logger.debug(f"list_repos returning {len(result)} repos: {[r['name'] for r in result]}")
    return result


def _collect_file_entries(
    root: Path,
    base_root: Path,
    repo: str,
    max_depth: int,
    include_hidden: bool,
    max_entries: int,
) -> tuple[List[Dict[str, str]], bool]:
    """
    Collect file and directory entries from a repository.
    
    Returns:
        Tuple of (entries list, truncated flag)
    """
    base_parts = len(root.relative_to(base_root).parts)
    entries: List[Dict[str, str]] = []
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root):
        if len(entries) >= max_entries:
            truncated = True
            break

        current = Path(dirpath)
        rel_parts = len(current.relative_to(base_root).parts)
        depth = rel_parts - base_parts

        if depth > max_depth:
            dirnames[:] = []
            continue

        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for d in dirnames:
            if len(entries) >= max_entries:
                truncated = True
                break
            rel_path = (current / d).relative_to(base_root).as_posix()
            entries.append({"repo": repo, "path": rel_path, "type": "dir"})

        if truncated:
            break

        for f in filenames:
            if len(entries) >= max_entries:
                truncated = True
                break
            if not include_hidden and f.startswith("."):
                continue
            rel_path = (current / f).relative_to(base_root).as_posix()
            entries.append({"repo": repo, "path": rel_path, "type": "file"})

        if truncated:
            break

    return entries, truncated


def list_repo_files(
    repo: str,
    subdir: str = ".",
    max_depth: int = 4,
    include_hidden: bool = False,
    max_entries: int = 1000,
) -> Dict[str, object]:
    """
    List files and directories under a subdirectory of a given repo.

    Args:
      repo: Logical repo name (as configured in config.json or SIGIL_REPO_MAP).
      subdir: Path relative to that repo root (e.g. "src", "crates/codex/src").
      max_depth: Maximum depth below `subdir` to traverse.
      include_hidden: Whether to include dotfiles / dot-directories.
      max_entries: Maximum number of entries to return (default: 1000).
                   Prevents timeouts on large repositories.

    Returns:
      A dictionary with:
        - entries: List of entries with "repo", "path", and "type" fields
        - total_found: Total number of entries found (may be > len(entries) if truncated)
        - truncated: True if results were truncated due to max_entries limit
    """
    logger.info(
        "list_repo_files tool called "
        "(repo=%r, subdir=%r, max_depth=%r, include_hidden=%r, max_entries=%r)",
        repo,
        subdir,
        max_depth,
        include_hidden,
        max_entries,
    )
    _ensure_repos_configured()

    root = _resolve_under_repo(repo, subdir)
    base_root = _get_repo_root(repo)

    # Ensure root is a Path object
    if not isinstance(root, Path):
        root = Path(root)
    if not isinstance(base_root, Path):
        base_root = Path(base_root)

    entries, truncated = _collect_file_entries(
        root, base_root, repo, max_depth, include_hidden, max_entries
    )

    if truncated:
        logger.warning(
            f"list_repo_files truncated at {max_entries} entries "
            f"for repo={repo}, subdir={subdir}"
        )

    # Sort dirs before files, then by repo, then by path
    entries.sort(key=lambda e: (e["type"], e["repo"], e["path"]))
    
    total_found = len(entries)
    
    logger.debug(
        f"list_repo_files returning {len(entries)} entries "
        f"(truncated={truncated}, first 5: {entries[:5] if entries else []})"
    )
    
    return {
        "entries": entries,
        "total_found": total_found,
        "truncated": truncated,
    }


@_safe_tool_decorator(
    title=(
        "Read-only 鈥� read_repo_file"
    ),
    description=(
        "Read a single file from a repository (read-only)."
        " Args: repo (repo name), path (relative path), max_bytes (defensive limit)."
    ),
    annotations={
        "operation": "read",
        "safety": "non-destructive",
        "audience": ["assistant", "user"],
    },
    inputSchema={
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "path": {"type": "string"},
            "max_bytes": {"type": "integer"}
        },
        "required": ["repo", "path"]
    },
)
def read_repo_file(
    repo: str,
    path: str,
    max_bytes: int = 20_000,
) -> str:
    """
    Read a single file from a given repo.

    Args:
      repo: Logical repo name (as defined in SIGIL_REPO_MAP).
      path: File path relative to that repo root (e.g. "src/main.rs").
      max_bytes: Maximum number of bytes to return (defensive limit).

    Returns:
      The file contents as a UTF-8 string (possibly truncated, with a notice).
    """
    logger.info(
        "read_repo_file tool called (repo=%r, path=%r, max_bytes=%r)",
        repo,
        path,
        max_bytes,
    )
    _ensure_repos_configured()

    file_path = _resolve_under_repo(repo, path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Path {file_path} is not a file or does not exist in repo {repo!r}."
        )

    data = file_path.read_bytes()
    if len(data) > max_bytes:
        return (
            data[:max_bytes].decode("utf-8", errors="replace")
            + "\n\n[... truncated ...]"
        )

    return data.decode("utf-8", errors="replace")


@_safe_tool_decorator(
    title=(
        "Read-only 鈥� search_repo"
    ),
    description=(
        "Naive full-text search across one repo or all repos (read-only)."
        " Args: query, optional repo, file_glob, max_results."
    ),
    annotations={
        "operation": "read",
        "safety": "non-destructive",
        "audience": ["assistant", "user"],
    },
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "repo": {"type": ["string", "null"]},
            "file_glob": {"type": "string"},
            "max_results": {"type": "integer"}
        },
        "required": ["query"]
    },
)
def search_repo(
    query: str,
    repo: Optional[str] = None,
    file_glob: str = "*.rs",
    max_results: int = 50,
) -> List[Dict[str, object]]:
    """
    Naive full-text search across one repo or all repos.

    Args:
      query: Substring to search for.
      repo: Logical repo name. If omitted or null, search all repos.
      file_glob: Glob pattern (e.g. "*.rs", "*.toml", "*").
      max_results: Stop after this many matches.

    Returns:
      A list of matches:
      {
        "repo": "<repo_name>",
        "path": "src/main.rs",
        "line": 42,
        "text": "the line content"
      }
    """
    logger.info(
        "search_repo tool called (query=%r, repo=%r, file_glob=%r, max_results=%r)",
        query,
        repo,
        file_glob,
        max_results,
    )
    _ensure_repos_configured()

    matches: List[Dict[str, object]] = []

    if repo is None:
        targets = REPOS
    else:
        # Raises ValueError if repo unknown
        targets = {repo: _get_repo_root(repo)}

    for repo_name, repo_root in targets.items():
        # Ensure repo_root is a Path object
        if not isinstance(repo_root, Path):
            repo_root = Path(repo_root)
        
        # rglob relative to root
        for path in repo_root.rglob(file_glob):
            if not path.is_file():
                continue

            # Skip hidden files/dirs unless explicitly using "*"
            parts = path.relative_to(repo_root).parts
            if file_glob != "*" and any(part.startswith(".") for part in parts):
                continue

            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.warning("Failed to read %s: %s", path, e)
                continue

            for idx, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    matches.append(
                        {
                            "repo": repo_name,
                            "path": path.relative_to(repo_root).as_posix(),
                            "line": idx,
                            "text": line.strip(),
                        }
                    )
                    if len(matches) >= max_results:
                        return matches

    return matches


@_safe_tool_decorator(
    title=(
        "Read-only 鈥� search"
    ),
    description=(
        "Deep-Research compatible search (read-only). Returns structured result list"
        " of id/title/url entries."
        " Args: query, optional repo, file_glob, max_results."
    ),
    annotations={
        "operation": "read",
        "safety": "non-destructive",
        "audience": ["assistant", "user"],
    },
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "repo": {"type": ["string", "null"]},
            "file_glob": {"type": "string"},
            "max_results": {"type": "integer"}
        },
        "required": ["query"]
    },
)
def search(
    query: str,
    repo: Optional[str] = None,
    file_glob: str = "*",
    max_results: int = 50,
) -> Dict[str, object]:
    """
    Deep Research-compatible search tool.

    Returns:
      {
        "results": [
          { "id": "...", "title": "...", "url": "..." },
          ...
        ]
      }
    """
    # Reuse existing search_repo logic
    raw_matches = search_repo(
        query=query,
        repo=repo,
        file_glob=file_glob,
        max_results=max_results,
    )

    results: List[Dict[str, str]] = []

    for m in raw_matches:
        repo_name = str(m["repo"])
        path = str(m["path"])
        line = int(m["line"])

        doc_id = f"{repo_name}::{path}"
        title = f"{repo_name}:{path} (line {line})"

        # URL can be any stable, citeable handle; for now we use a pseudo-URL
        url = f"mcp://sigil_repos/{doc_id}"

        results.append(
            {
                "id": doc_id,
                "title": title,
                "url": url,
            }
        )

    return {"results": results}


def fetch(doc_id: str) -> Dict[str, object]:
    """
    Deep Research-compatible fetch tool.

    Args:
      doc_id: A string of the form "<repo>::<relative/path>".

    Returns:
      {
        "id": doc_id,
        "title": "...",
        "text": "file contents...",
        "url": "mcp://sigil_repos/<doc_id>",
        "metadata": {...}
      }
    """
    _ensure_repos_configured()

    if "::" not in doc_id:
        raise ValueError(
            f"Invalid doc_id {doc_id!r}; expected '<repo>::<relative/path>'"
        )

    repo, rel_path = doc_id.split("::", 1)
    text = read_repo_file(repo=repo, path=rel_path, max_bytes=50_000)

    title = f"{repo}:{rel_path}"
    url = f"mcp://sigil_repos/{doc_id}"

    return {
        "id": doc_id,
        "title": title,
        "text": text,
        "url": url,
        "metadata": {
            "repo": repo,
            "path": rel_path,
        },
    }


# --------------------------------------------------------------------
# IDE-like indexing tools for ChatGPT integration
# --------------------------------------------------------------------


def index_repository(
    repo: str,
    force_rebuild: bool = False
) -> Dict[str, object]:
    """
    Build or rebuild the search index for a repository.
    
    This enables fast code search and IDE-like features (go-to-definition,
    symbol search, file outline). The index includes both text search
    (trigrams) and semantic information (symbols extracted via ctags).
    
    Args:
      repo: Logical repo name (as defined in SIGIL_REPO_MAP)
      force_rebuild: If true, rebuild index from scratch (default: false)
    
    Returns:
      Statistics about the indexing operation:
      - files_indexed: Number of files processed
      - symbols_extracted: Number of code symbols found (functions, classes, etc.)
      - trigrams_built: Number of trigram entries for text search
      - bytes_indexed: Total bytes processed
      - duration_seconds: Time taken to build index
    
    Example:
      To index the 'runtime' repository:
      index_repository(repo="runtime")
      
      To force a full rebuild:
      index_repository(repo="runtime", force_rebuild=True)
    """
    logger.info(
        "index_repository tool called (repo=%r, force_rebuild=%r)",
        repo,
        force_rebuild
    )
    _ensure_repos_configured()
    
    repo_path = _get_repo_root(repo)
    index = _get_index()
    
    stats = index.index_repository(repo, repo_path, force=force_rebuild)
    
    return {
        "status": "completed",
        "repo": repo,
        **stats
    }


@_safe_tool_decorator(
    title=(
        "Read-only 鈥� search_code"
    ),
    description=(
        "Fast indexed code search (read-only) using trigram indexing. Returns"
        " filenames, line numbers and matching text."
        " Args: query, optional repo, max_results."
    ),
    annotations={
        "operation": "read",
        "safety": "non-destructive",
        "audience": ["assistant", "user"],
    },
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "repo": {"type": ["string", "null"]},
            "max_results": {"type": "integer"}
        },
        "required": ["query"]
    },
)
def search_code(
    query: str,
    repo: Optional[str] = None,
    max_results: int = 50
) -> List[Dict[str, object]]:
    """
    Fast indexed code search across repositories.
    
    Uses trigram-based indexing for substring search, much faster than
    grep-style search. Returns results with line numbers and context.
    
    Args:
      query: Text to search for (case-insensitive substring match)
      repo: Optional repo name to restrict search (searches all repos if omitted)
      max_results: Maximum number of results to return (default: 50)
    
    Returns:
      List of matches, each containing:
      - repo: Repository name
      - path: File path within repository
      - line: Line number where match was found
      - text: The matching line of code
      - doc_id: Document ID for fetching full file (use with fetch tool)
    
    Example:
      Search for "async def" across all repositories:
      search_code(query="async def")
      
      Search only in the 'runtime' repository:
      search_code(query="async def", repo="runtime")
    """
    logger.info(
        "search_code tool called (query=%r, repo=%r, max_results=%r)",
        query,
        repo,
        max_results
    )
    _ensure_repos_configured()
    
    index = _get_index()
    results = index.search_code(query, repo=repo, max_results=max_results)
    
    return [
        {
            "repo": r.repo,
            "path": r.path,
            "line": r.line,
            "text": r.text,
            "doc_id": r.doc_id
        }
        for r in results
    ]


def goto_definition(
    symbol_name: str,
    repo: Optional[str] = None,
    kind: Optional[str] = None
) -> List[Dict[str, object]]:
    """
    Find where a symbol is defined (IDE "Go to Definition" feature).
    
    Searches the symbol index to find definitions of functions, classes,
    methods, variables, etc. This provides semantic search beyond simple
    text matching.
    
    Args:
      symbol_name: Name of the symbol to find (e.g., "MyClass", "process_data")
      repo: Optional repo name to restrict search
      kind: Optional symbol type filter:
            - "function" or "f": Functions/procedures
            - "class" or "c": Classes/types
            - "method" or "m": Class methods
            - "variable" or "v": Variables
            - Other values: member, macro, struct, enum, etc.
    
    Returns:
      List of symbol definitions, each containing:
      - name: Symbol name
      - kind: Symbol type (function, class, method, etc.)
      - file_path: Location as "repo::path"
      - line: Line number where defined
      - signature: Function/method signature (if available)
      - scope: Containing scope like class name (if available)
    
    Example:
      Find where "HttpClient" class is defined:
      goto_definition(symbol_name="HttpClient", kind="class")
      
      Find all definitions of "process" function:
      goto_definition(symbol_name="process", kind="function")
    """
    logger.info(
        "goto_definition tool called (symbol_name=%r, repo=%r, kind=%r)",
        symbol_name,
        repo,
        kind
    )
    _ensure_repos_configured()
    
    index = _get_index()
    symbols = index.find_symbol(symbol_name, kind=kind, repo=repo)
    
    return [
        {
            "name": s.name,
            "kind": s.kind,
            "file_path": s.file_path,
            "line": s.line,
            "signature": s.signature,
            "scope": s.scope
        }
        for s in symbols
    ]


def list_symbols(
    repo: str,
    file_path: Optional[str] = None,
    kind: Optional[str] = None
) -> List[Dict[str, object]]:
    """
    List symbols in a file or repository (IDE "Outline" or "Structure" view).
    
    Shows an overview of code structure including functions, classes, methods,
    and other symbols. Useful for understanding a file's contents or getting
    a high-level view of a codebase.
    
    Args:
      repo: Repository name
      file_path: Optional file path to show symbols for (relative to repo root)
                 If omitted, shows symbols from entire repository
      kind: Optional symbol type filter (function, class, method, etc.)
    
    Returns:
      List of symbols sorted by file path and line number, each containing:
      - name: Symbol name
      - kind: Symbol type
      - file_path: File path within repository
      - line: Line number
      - signature: Function/method signature (if available)
      - scope: Containing scope (if available)
    
    Example:
      Show all symbols in a specific file:
      list_symbols(repo="runtime", file_path="src/main.rs")
      
      Show all functions in the runtime repository:
      list_symbols(repo="runtime", kind="function")
      
      Show all classes across the repository:
      list_symbols(repo="runtime", kind="class")
    """
    logger.info(
        "list_symbols tool called (repo=%r, file_path=%r, kind=%r)",
        repo,
        file_path,
        kind
    )
    _ensure_repos_configured()
    
    index = _get_index()
    symbols = index.list_symbols(repo, file_path=file_path, kind=kind)
    
    return [
        {
            "name": s.name,
            "kind": s.kind,
            "file_path": s.file_path,
            "line": s.line,
            "signature": s.signature,
            "scope": s.scope
        }
        for s in symbols
    ]


def get_index_stats(repo: Optional[str] = None) -> Dict[str, object]:
    """
    Get statistics about the code index.
    
    Shows information about indexed repositories, number of files,
    symbols extracted, and when the index was last updated.
    
    Args:
      repo: Optional repo name to get stats for specific repository
            If omitted, returns global statistics across all repos
    
    Returns:
      For specific repo:
      - repo: Repository name
      - documents: Number of indexed files
      - symbols: Number of extracted symbols
      - indexed_at: Timestamp of last indexing
      
      For all repos:
      - repositories: Number of indexed repositories
      - documents: Total number of indexed files
      - symbols: Total number of symbols
      - trigrams: Number of trigram index entries
    
    Example:
      Get global stats:
      get_index_stats()
      
      Get stats for specific repo:
      get_index_stats(repo="runtime")
    """
    logger.info("get_index_stats tool called (repo=%r)", repo)
    _ensure_repos_configured()
    
    index = _get_index()
    stats = index.get_index_stats(repo=repo)
    # Cast to Dict[str, object] to satisfy type checker
    return {k: v for k, v in stats.items()}


def build_vector_index(
    repo: str,
    force_rebuild: bool = False,
    model: str = "default",
) -> Dict[str, object]:
    """
    Build or refresh the vector (semantic) index for a repository.
    
    This computes embeddings for code chunks and stores them in the index
    for fast semantic search.
    
    Args:
      repo: Repository name to index
      force_rebuild: If True, rebuild all embeddings (default: False)
      model: Embedding model identifier (default: "default")
    
    Returns:
      Statistics about the indexing operation:
      - status: "completed"
      - repo: Repository name
      - model: Model identifier used
      - chunks_indexed: Number of code chunks embedded
      - documents_processed: Number of files processed
    
    Example:
      Build vector index for 'runtime' repository:
      build_vector_index(repo="runtime")
      
      Force rebuild with custom model:
      build_vector_index(repo="runtime", force_rebuild=True, model="custom-model")
    """
    logger.info(
        "build_vector_index called (repo=%r, force_rebuild=%r, model=%r)",
        repo,
        force_rebuild,
        model,
    )
    _ensure_repos_configured()
    
    index = _get_index()
    
    # Ensure the basic index exists first
    repo_path = _get_repo_root(repo)
    index.index_repository(repo, repo_path, force=False)
    
    stats = index.build_vector_index(
        repo=repo,
        embed_fn=index.embed_fn,
        model=model,
        force=force_rebuild,
    )
    
    return {
        "status": "completed",
        "repo": repo,
        "model": model,
        **stats,
    }


@_safe_tool_decorator(
    title=(
        "Read-only 鈥� semantic_search"
    ),
    description=(
        "Semantic code search using vector embeddings (read-only)."
        " Args: query (natural language or code-like query), optional repo filter,"
        " k (top-k), model (embedding model id)."
    ),
    annotations={
        "operation": "read",
        "safety": "non-destructive",
        "audience": ["assistant", "user"],
    },
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "repo": {"type": "string"},
            "k": {"type": "integer"},
            "model": {"type": "string"},
            "code_only": {"type": "boolean"},
            "prefer_code": {"type": "boolean"}
        },
        "required": ["query"]
    },
)
def semantic_search(
    query: str,
    repo: Optional[str] = None,
    k: int = 20,
    model: str = "default",
    code_only: bool = False,
    prefer_code: bool = False,
) -> Dict[str, object]:
    """
    Semantic code search using vector embeddings.
    
    Search for code based on meaning and intent rather than exact text matching.
    Uses vector embeddings to find semantically similar code chunks.
    
    Args:
      query: Natural language or code-like query describing what you're looking for
      repo: Repository name (optional; when omitted searches across all indexed repos)
      k: Number of results to return (default: 20)
      model: Embedding model identifier (default: "default")
      code_only: When true, hard-filter results to code chunks
      prefer_code: When true, rerank to favor code while still allowing docs/config
    
    Returns:
      {
        "matches": [
          {
            "repo": "...",
            "path": "src/main.rs",
            "start_line": 10,
            "end_line": 110,
            "score": 0.83,
            "doc_id": "repo::src/main.rs"
          },
          ...
        ]
      }
    
    Example:
      Find authentication-related code:
      semantic_search(
          query="user authentication and login handlers",
          repo="runtime"
      )
      
      Find error handling code:
      semantic_search(
          query="error handling middleware",
          repo="runtime",
          k=10
      )
    """
    logger.info(
        "semantic_search called (query=%r, repo=%r, k=%r, model=%r, code_only=%r, prefer_code=%r)",
        query,
        repo,
        k,
        model,
        code_only,
        prefer_code,
    )
    _ensure_repos_configured()

    index = _get_index()

    # If embeddings are not configured, fail gracefully instead of crashing
    if index.embed_fn is None or index.vectors is None:
        logger.warning(
            "semantic_search called but embeddings are not configured "
            "(embeddings_enabled=%r, embed_model=%r)",
            getattr(config, "embeddings_enabled", None),
            getattr(index, "embed_model", None),
        )
        return {
            "status": "error",
            "error": (
                "Semantic search is not available because embeddings are not "
                "configured on the server. See docs/EMBEDDING_SETUP.md."
            ),
        }

    # Map the default sentinel to the index's configured model, if any
    effective_model = None if model == "default" else model

    try:
        matches = index.semantic_search(
            query=query,
            repo=repo,
            k=k,
            embed_fn=index.embed_fn,
            model=effective_model,
            code_only=code_only,
            prefer_code=prefer_code,
        )
    except Exception as exc:  # pragma: no cover - defensive catch for MCP stability
        logger.exception("semantic_search failed: %s", exc)
        return {
            "status": "error",
            "error": f"semantic_search failed: {exc}",
        }
    
    return {
        "status": "completed",
        "repo": repo,
        "model": effective_model or index.embed_model,
        "matches": matches,
    }


def _setup_authentication():
    """Initialize and log authentication configuration."""
    if AUTH_ENABLED:
        logger.info("=" * 60)
        logger.info("AUTHENTICATION ENABLED")
        logger.info("=" * 60)
        
        # Initialize OAuth if enabled
        if OAUTH_ENABLED:
            logger.info("")
            logger.info("馃攼 OAuth2 Authentication")
            logger.info("=" * 60)
            
            oauth_manager = get_oauth_manager()
            credentials = oauth_manager.initialize_client()
            
            if credentials:
                client_id, client_secret = credentials
                logger.info("馃啎 NEW OAuth client created!")
                logger.info("")
                logger.info(f"Client ID:     {client_id}")
                logger.info(f"Client Secret: {client_secret}")
                logger.info("")
                logger.info("[WARNING]  SAVE THESE CREDENTIALS SECURELY!")
                logger.info("")
            else:
                client = oauth_manager.get_client()
                if client:
                    logger.info(f"Using existing OAuth client: {client.client_id}")
                    logger.info("(Client secret stored securely)")
                logger.info("")
            
            logger.info("OAuth Endpoints:")
            logger.info("  - Authorization: /oauth/authorize")
            logger.info("  - Token:         /oauth/token")
            logger.info("  - Revoke:        /oauth/revoke")
            logger.info("")
        
        # Initialize API key
        api_key = initialize_api_key()
        
        if api_key:
            logger.info(" NEW API Key Generated")
            logger.info("=" * 60)
            logger.info(f"API Key: {api_key}")
            logger.info("=" * 60)
            logger.info("")
            logger.info("[WARNING]  This is the ONLY time you'll see this key!")
            logger.info("   Set it in your environment:")
            logger.info(f"   export SIGIL_MCP_API_KEY={api_key}")
            logger.info("")
        else:
            logger.info("Using existing API key from ~/.sigil_mcp_server/api_key")
            logger.info("(Fallback for local development)")
            logger.info("")
        
        if ALLOW_LOCAL_BYPASS:
            logger.info("[YES] Local connections (127.0.0.1) bypass authentication")
            logger.info("")
        
        if ALLOWED_IPS:
            logger.info(f"IP Whitelist enabled: {', '.join(ALLOWED_IPS)}")
            logger.info("")
    else:
        logger.warning("=" * 60)
        logger.warning("[WARNING]  AUTHENTICATION DISABLED")
        logger.warning("=" * 60)
        logger.warning("To enable authentication, set:")
        logger.warning("  export SIGIL_MCP_AUTH_ENABLED=true")
        logger.warning("")


def _setup_file_watching():
    """Start file watching if enabled."""
    if config.watch_enabled and REPOS:
        logger.info("=" * 60)
        logger.info("[INFO] FILE WATCHING ENABLED")
        logger.info("=" * 60)
        logger.info(f"Debounce: {config.watch_debounce_seconds}s")
        logger.info("Watching repositories for changes...")
        _start_watching_repos()
        logger.info("")


def _build_sse_app(*, sse_route_path: str, message_route_path: str):
    """Construct an SSE transport app for FastMCP with optional bearer gating."""
    middleware: list[Middleware] = []
    # Require token only when configured; allow local bypass if enabled globally
    if config.mcp_require_token or config.mcp_server_token:
        if config.mcp_require_token and not config.mcp_server_token:
            logger.warning(
                "MCP bearer auth required but no token configured; SSE requests will be rejected"
            )
        middleware.append(
            Middleware(
                MCPBearerAuthMiddleware,
                token=config.mcp_server_token,
                require_token=config.mcp_require_token,
                allow_local_bypass=ALLOW_LOCAL_BYPASS,
            )
        )

    auth_provider = getattr(mcp, "_auth_server_provider", None)
    additional_routes = getattr(mcp, "_additional_http_routes", None)
    if not hasattr(mcp, "_get_additional_http_routes"):
        # FastMCP versions prior to exposing _get_additional_http_routes require a shim
        setattr(mcp, "_get_additional_http_routes", lambda: [])  # type: ignore[attr-defined]

    return create_sse_app(
        server=mcp,
        message_path=message_route_path,
        sse_path=sse_route_path,
        auth=auth_provider,
        debug=mcp.settings.debug,
        routes=additional_routes,
        middleware=middleware,
    )


def build_parent_app(include_admin: Optional[bool] = None) -> Starlette:
    """
    Create parent ASGI app that combines FastMCP, optional Admin API, and SSE transport.
    """
    include_admin = config.admin_enabled if include_admin is None else include_admin

    _initialize_external_mcp()

    # Get FastMCP's ASGI app (streamable HTTP) and session manager
    mcp_asgi_app = mcp.streamable_http_app()
    session_manager = mcp.session_manager

    from starlette.routing import BaseRoute

    parent_routes: list[BaseRoute] = []

    if include_admin:
        logger.info("=" * 60)
        logger.info("[INFO] ADMIN API ENABLED")
        logger.info("=" * 60)
        logger.info("Admin API integrated into main server process")
        logger.info("Both MCP and Admin API share the same index instance")
        logger.info("Admin endpoints available at /admin/*")
        logger.info("")
        admin_app = _get_admin_app()
        parent_routes.extend(admin_app.routes)

    # Mount SSE transport at its parent prefix; use relative paths inside the mounted app
    sse_path = PurePosixPath(config.mcp_sse_path)
    message_path = PurePosixPath(config.mcp_message_path)
    mount_prefix = str(sse_path.parent) or "/"
    if not mount_prefix.startswith("/"):
        mount_prefix = "/" + mount_prefix
    try:
        rel_message_path = "/" + str(message_path.relative_to(mount_prefix)).lstrip("/")
    except ValueError:
        rel_message_path = "/" + message_path.name
    rel_sse_path = "/" + sse_path.name

    sse_app = _build_sse_app(
        sse_route_path=rel_sse_path,
        message_route_path=rel_message_path,
    )
    parent_routes.append(Mount(mount_prefix, app=sse_app))

    # Mount FastMCP streamable HTTP transport; preserve legacy root mount
    parent_routes.append(Mount(config.mcp_http_path, app=mcp_asgi_app))
    if config.mcp_http_path != "/":
        parent_routes.append(Mount("/", app=mcp_asgi_app))

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def combined_lifespan(app: Starlette):
        async with session_manager.run():
            yield

    parent_app = Starlette(routes=parent_routes, lifespan=combined_lifespan)

    if include_admin:
        from starlette.middleware.cors import CORSMiddleware

        parent_app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ],
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    wrapped_parent = HeaderLoggingASGIMiddleware(parent_app)
    return wrapped_parent


def main():
    """Main entry point for the Sigil MCP Server."""
    logger.info("Starting sigil_repos MCP server (transport=streamable-http)")

    _setup_authentication()
    _setup_file_watching()
    start_admin_ui()

    import uvicorn

    try:
        uvicorn.run(
            build_parent_app(include_admin=config.admin_enabled),
            host=config.server_host,
            port=config.server_port,
            log_level=config.log_level.lower(),
        )
    finally:
        stop_admin_ui()
        if _WATCHER:
            logger.info("Stopping file watcher...")
            _WATCHER.stop()


if __name__ == "__main__":
    main()
