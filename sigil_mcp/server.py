# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import logging
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlencode
import numpy as np
import os

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from .indexer import SigilIndex
from .auth import initialize_api_key, verify_api_key, get_api_key_from_env
from .oauth import get_oauth_manager
from .config import get_config
from .watcher import FileWatchManager

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

config = get_config()

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------

logger = logging.getLogger("sigil_repos_mcp")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(config.log_level)

# --------------------------------------------------------------------
# Security Configuration
# --------------------------------------------------------------------

# Load from config (falls back to environment variables)
AUTH_ENABLED = config.auth_enabled
OAUTH_ENABLED = config.oauth_enabled
ALLOW_LOCAL_BYPASS = config.allow_local_bypass
ALLOWED_IPS = config.allowed_ips

# --------------------------------------------------------------------
# MCP server
# --------------------------------------------------------------------

# Disable ALL transport security for ChatGPT compatibility
# ChatGPT's MCP connector sends:
# 1. Content-Type: application/octet-stream (invalid, should be application/json)
# 2. Host headers that don't match localhost (ngrok domains)
# We must explicitly disable DNS rebinding protection to accept these requests
transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False
)

mcp = FastMCP(
    name=config.server_name, 
    json_response=True,
    streamable_http_path="/",
    transport_security=transport_security
)


# --------------------------------------------------------------------
# Authentication Middleware
# --------------------------------------------------------------------

def is_local_connection(client_ip: str = None) -> bool:
    """
    Check if connection is from localhost.
    
    Args:
        client_ip: Client IP address
    
    Returns:
        True if localhost, False otherwise
    """
    if not client_ip:
        return False
    
    local_ips = {"127.0.0.1", "::1", "localhost"}
    return client_ip in local_ips


def check_authentication(
    request_headers: Dict[str, str] = None,
    client_ip: str = None
) -> bool:
    """
    Check if request is authenticated.
    
    Args:
        request_headers: HTTP request headers (if available)
        client_ip: Client IP address
    
    Returns:
        True if authenticated or auth disabled, False otherwise
    """
    # Allow local connections without auth if enabled
    if ALLOW_LOCAL_BYPASS and is_local_connection(client_ip):
        logger.debug("Local connection - bypassing authentication")
        return True
    
    if not AUTH_ENABLED:
        return True
    
    # Check for OAuth token first (preferred)
    if OAUTH_ENABLED and request_headers:
        auth_header = (
            request_headers.get("Authorization")
            or request_headers.get("authorization")
        )
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
                oauth_manager = get_oauth_manager()
                if oauth_manager.verify_token(token):
                    logger.debug("OAuth token valid")
                    return True
    
    # Fall back to API key authentication
    env_key = get_api_key_from_env()
    if env_key and verify_api_key(env_key):
        return True
    
    if request_headers:
        api_key = (
            request_headers.get("x-api-key") or
            request_headers.get("X-API-Key")
        )
        
        if api_key and verify_api_key(api_key):
            return True
    
    logger.warning("Authentication failed - no valid credentials provided")
    return False


def check_ip_whitelist(client_ip: str = None) -> bool:
    """
    Check if client IP is whitelisted.
    
    Args:
        client_ip: Client IP address
    
    Returns:
        True if IP is allowed or whitelist is empty, False otherwise
    """
    if not ALLOWED_IPS or not ALLOWED_IPS[0]:
        return True
    
    if client_ip in ALLOWED_IPS:
        return True
    
    logger.warning(f"IP {client_ip} not in whitelist")
    return False

# --------------------------------------------------------------------
# Repo configuration
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Repository Configuration
# --------------------------------------------------------------------

# Load repositories from config
REPOS: Dict[str, Path] = config.repositories

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


def _create_embedding_function():
    """
    Create embedding function based on configuration.
    
    Returns:
        Tuple of (embed_fn, model_name) or (None, None) if embeddings disabled
    """
    if not config.embeddings_enabled:
        logger.info("Embeddings disabled in config")
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
        def embed_fn(texts: List[str]) -> np.ndarray:
            embeddings_list = embedding_provider.embed_documents(texts)
            return np.array(embeddings_list, dtype="float32")
        
        model_name = f"{provider}:{model}"
        logger.info(f"Embeddings initialized: {model_name} (dim={dimension})")
        return embed_fn, model_name
        
    except ImportError as e:
        logger.error(
            f"Failed to import embedding provider '{provider}': {e}. "
            "Install required dependencies. See docs/EMBEDDING_SETUP.md. "
            "Embeddings disabled."
        )
        return None, None
    except Exception as e:
        logger.error(
            f"Failed to initialize embedding provider '{provider}': {e}. "
            "Embeddings disabled."
        )
        return None, None


def _get_index() -> SigilIndex:
    """Get or create the global index instance."""
    global _INDEX
    if _INDEX is None:
        index_path = config.index_path
        embed_fn, embed_model = _create_embedding_function()
        _INDEX = SigilIndex(index_path, embed_fn=embed_fn, embed_model=embed_model)
        logger.info(f"Initialized index at {index_path}")
    return _INDEX


def _on_file_change(repo_name: str, file_path: Path, event_type: str):
    """Handle file change events from watcher."""
    logger.info(f"File {event_type}: {file_path.name} in {repo_name}")
    
    try:
        index = _get_index()
        repo_path = _get_repo_root(repo_name)
        
        if event_type == "deleted":
            # For deletions, we'd need to remove from index
            # For now, just log it (full re-index will handle cleanup)
            logger.info(f"File deleted: {file_path.name} (index cleanup deferred)")
        else:
            # Granular re-indexing for modified/created files
            success = index.index_file(repo_name, repo_path, file_path)
            if success:
                logger.info(f"Re-indexed {file_path.name} after {event_type}")
            else:
                logger.debug(f"Skipped re-indexing {file_path.name}")
    except Exception as e:
        logger.error(f"Error re-indexing after {event_type}: {e}")


def _get_watcher() -> Optional[FileWatchManager]:
    """Get or create the global file watcher."""
    global _WATCHER
    
    if not config.watch_enabled:
        return None
    
    if _WATCHER is None:
        _WATCHER = FileWatchManager(
            on_change=_on_file_change,
            ignore_dirs=config.watch_ignore_dirs,
            ignore_extensions=config.watch_ignore_extensions,
        )
        _WATCHER.start()
        logger.info("File watcher initialized")
    
    return _WATCHER


def _start_watching_repos():
    """Start watching all configured repositories."""
    watcher = _get_watcher()
    if watcher:
        for repo_name, repo_path in REPOS.items():
            watcher.watch_repository(repo_name, Path(repo_path))


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


@mcp.custom_route("/oauth/authorize", methods=["GET", "POST"])
async def oauth_authorize_http(request: Request) -> JSONResponse | RedirectResponse:
    """OAuth 2.0 Authorization Endpoint."""
    logger.info("OAuth authorization request received")
    
    if not OAUTH_ENABLED:
        return JSONResponse({"error": "oauth_not_enabled"}, status_code=501)
    
    # Get parameters from query string or form data
    if request.method == "GET":
        params = dict(request.query_params)
    else:
        form = await request.form()
        params = dict(form)
    
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
    # Allow any HTTPS redirect URI or registered URIs for flexibility with ChatGPT
    if not (redirect_uri in client.redirect_uris or redirect_uri.startswith("https://")):
        return JSONResponse({
            "error": "invalid_request",
            "error_description": "Redirect URI must be HTTPS or registered"
        }, status_code=400)
    
    # Auto-approve (for trusted clients)
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
    
    return RedirectResponse(redirect_url, status_code=302)


@mcp.custom_route("/oauth/token", methods=["POST"])
async def oauth_token_http(request: Request) -> JSONResponse:
    """OAuth 2.0 Token Endpoint."""
    logger.info("OAuth token request received")
    
    if not OAUTH_ENABLED:
        return JSONResponse({"error": "oauth_not_enabled"}, status_code=501)
    
    # Parse form data
    form = await request.form()
    grant_type = form.get("grant_type")
    code = form.get("code")
    redirect_uri = form.get("redirect_uri")
    client_id = form.get("client_id")
    client_secret = form.get("client_secret")
    code_verifier = form.get("code_verifier")
    refresh_token = form.get("refresh_token")
    
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
    token = form.get("token")
    client_id = form.get("client_id")
    client_secret = form.get("client_secret")
    
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


@mcp.tool()
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
    if redirect_uri not in client.redirect_uris:
        return {
            "error": "invalid_request",
            "error_description": "Redirect URI not registered for this client"
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
    
    return {
        "redirect_url": redirect_url,
        "code": code,
        "state": state
    }


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def list_repos() -> List[Dict[str, str]]:
    """
    List all configured repositories.

    Each entry has:
      - name: Logical repo name
      - path: Absolute filesystem path
    """
    logger.info("list_repos tool called")
    _ensure_repos_configured()

    return [
        {"name": name, "path": str(path)}
        for name, path in sorted(REPOS.items(), key=lambda kv: kv[0])
    ]


@mcp.tool()
def list_repo_files(
    repo: str,
    subdir: str = ".",
    max_depth: int = 4,
    include_hidden: bool = False,
) -> List[Dict[str, str]]:
    """
    List files and directories under a subdirectory of a given repo.

    Args:
      repo: Logical repo name (as configured in config.json or SIGIL_REPO_MAP).
      subdir: Path relative to that repo root (e.g. "src", "crates/codex/src").
      max_depth: Maximum depth below `subdir` to traverse.
      include_hidden: Whether to include dotfiles / dot-directories.

    Returns:
      A list of entries: { "repo": "<repo>", "path": "src/main.rs", "type": "file"|"dir" }.
    """
    logger.info(
        "list_repo_files tool called (repo=%r, subdir=%r, max_depth=%r, include_hidden=%r)",
        repo,
        subdir,
        max_depth,
        include_hidden,
    )
    _ensure_repos_configured()

    root = _resolve_under_repo(repo, subdir)
    base_root = _get_repo_root(repo)

    # Depth of the starting directory relative to repo root
    base_parts = len(root.relative_to(base_root).parts)

    entries: List[Dict[str, str]] = []

    # Ensure root is a Path object
    if not isinstance(root, Path):
        root = Path(root)
    if not isinstance(base_root, Path):
        base_root = Path(base_root)

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        rel_parts = len(current.relative_to(base_root).parts)
        depth = rel_parts - base_parts

        if depth > max_depth:
            # Prevent walking deeper
            dirnames[:] = []
            continue

        # Filter hidden directories (if requested)
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        rel_dir = current.relative_to(base_root).as_posix()

        # Directories
        if rel_dir == ".":
            rel_dir = ""
        for d in dirnames:
            rel_path = (current / d).relative_to(base_root).as_posix()
            entries.append({"repo": repo, "path": rel_path, "type": "dir"})

        # Files
        for f in filenames:
            if not include_hidden and f.startswith("."):
                continue
            rel_path = (current / f).relative_to(base_root).as_posix()
            entries.append({"repo": repo, "path": rel_path, "type": "file"})

    # Sort dirs before files, then by repo, then by path
    entries.sort(key=lambda e: (e["type"], e["repo"], e["path"]))
    return entries


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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
    return index.get_index_stats(repo=repo)


@mcp.tool()
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


@mcp.tool()
def semantic_search(
    query: str,
    repo: str,
    k: int = 20,
    model: str = "default",
) -> Dict[str, object]:
    """
    Semantic code search using vector embeddings.
    
    Search for code based on meaning and intent rather than exact text matching.
    Uses vector embeddings to find semantically similar code chunks.
    
    Args:
      query: Natural language or code-like query describing what you're looking for
      repo: Repository name (required)
      k: Number of results to return (default: 20)
      model: Embedding model identifier (default: "default")
    
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
        "semantic_search called (query=%r, repo=%r, k=%r, model=%r)",
        query,
        repo,
        k,
        model,
    )
    _ensure_repos_configured()
    
    index = _get_index()
    matches = index.semantic_search(
        query=query,
        repo=repo,
        k=k,
        embed_fn=index.embed_fn,
        model=model,
    )
    
    return {"matches": matches}


def main():
    """Main entry point for the Sigil MCP Server."""
    logger.info("Starting sigil_repos MCP server (transport=streamable-http)")
    
    # Initialize authentication
    if AUTH_ENABLED:
        logger.info("=" * 60)
        logger.info("AUTHENTICATION ENABLED")
        logger.info("=" * 60)
        
        # Initialize OAuth if enabled
        if OAUTH_ENABLED:
            logger.info("")
            logger.info("🔐 OAuth2 Authentication")
            logger.info("=" * 60)
            
            oauth_manager = get_oauth_manager()
            credentials = oauth_manager.initialize_client()
            
            if credentials:
                client_id, client_secret = credentials
                logger.info("🆕 NEW OAuth client created!")
                logger.info("")
                logger.info(f"Client ID:     {client_id}")
                logger.info(f"Client Secret: {client_secret}")
                logger.info("")
                logger.info("⚠️  SAVE THESE CREDENTIALS SECURELY!")
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
            logger.info("🔑 NEW API Key Generated")
            logger.info("=" * 60)
            logger.info(f"API Key: {api_key}")
            logger.info("=" * 60)
            logger.info("")
            logger.info("⚠️  This is the ONLY time you'll see this key!")
            logger.info("   Set it in your environment:")
            logger.info(f"   export SIGIL_MCP_API_KEY={api_key}")
            logger.info("")
        else:
            logger.info("Using existing API key from ~/.sigil_mcp_server/api_key")
            logger.info("(Fallback for local development)")
            logger.info("")
        
        if ALLOW_LOCAL_BYPASS:
            logger.info("✅ Local connections (127.0.0.1) bypass authentication")
            logger.info("")
        
        if ALLOWED_IPS:
            logger.info(f"IP Whitelist enabled: {', '.join(ALLOWED_IPS)}")
            logger.info("")
    else:
        logger.warning("=" * 60)
        logger.warning("⚠️  AUTHENTICATION DISABLED")
        logger.warning("=" * 60)
        logger.warning("To enable authentication, set:")
        logger.warning("  export SIGIL_MCP_AUTH_ENABLED=true")
        logger.warning("")
    
    # Start file watching if enabled
    if config.watch_enabled and REPOS:
        logger.info("=" * 60)
        logger.info("📁 FILE WATCHING ENABLED")
        logger.info("=" * 60)
        logger.info(f"Debounce: {config.watch_debounce_seconds}s")
        logger.info("Watching repositories for changes...")
        _start_watching_repos()
        logger.info("")
    
    try:
        mcp.run(transport="streamable-http")
    finally:
        # Cleanup on shutdown
        if _WATCHER:
            logger.info("Stopping file watcher...")
            _WATCHER.stop()


if __name__ == "__main__":
    main()
