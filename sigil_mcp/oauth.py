# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
OAuth2 authentication for Sigil MCP Server.

Implements OAuth2 authorization code flow for secure remote access.
Local connections (localhost/127.0.0.1) can bypass authentication.
"""

import secrets
import hashlib
import json
import time
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

# OAuth configuration directory (override with SIGIL_MCP_OAUTH_DIR for tests)
def _resolve_oauth_dir() -> Path:
    base = os.getenv("SIGIL_MCP_OAUTH_DIR")
    if base:
        return Path(base).expanduser()
    return Path.home() / ".sigil_mcp_server" / "oauth"


OAUTH_DIR = _resolve_oauth_dir()
OAUTH_DIR.mkdir(parents=True, exist_ok=True)

# OAuth files
CLIENT_FILE = OAUTH_DIR / "client.json"
TOKENS_FILE = OAUTH_DIR / "tokens.json"
STATE_FILE = OAUTH_DIR / "state.json"


@dataclass
class OAuthClient:
    """OAuth client configuration."""
    client_id: str
    client_secret: str
    redirect_uris: list[str]
    created_at: int


@dataclass
class OAuthToken:
    """OAuth access token."""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    created_at: Optional[int] = None


@dataclass
class OAuthState:
    """OAuth authorization state for CSRF protection."""
    state: str
    code_verifier: Optional[str] = None
    redirect_uri: Optional[str] = None
    created_at: int = 0


class OAuthManager:
    """Manages OAuth2 authentication flow."""
    
    def __init__(self):
        self.tokens: Dict[str, OAuthToken] = {}
        self.states: Dict[str, OAuthState] = {}
        # authorization_code -> {client_id, redirect_uri, code_challenge, used}
        self.codes: Dict[str, Dict] = {}
        self._load_tokens()
    
    def initialize_client(self) -> Optional[Tuple[str, str]]:
        """
        Initialize OAuth client credentials.
        
        Returns:
            (client_id, client_secret) tuple if new client created, None if exists
        """
        if CLIENT_FILE.exists():
            logger.info("OAuth client already configured")
            return None
        
        client_id = f"sigil_{secrets.token_urlsafe(16)}"
        client_secret = secrets.token_urlsafe(32)
        
        # Allow common redirect URIs (ChatGPT, localhost, etc.)
        client = OAuthClient(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uris=[
                "http://localhost:8080/oauth/callback",
                "https://chatgpt.com/aip/oauth/callback",
                "https://chat.openai.com/aip/oauth/callback",
            ],
            created_at=int(time.time())
        )
        
        with open(CLIENT_FILE, 'w') as f:
            json.dump(asdict(client), f, indent=2)
        
        CLIENT_FILE.chmod(0o600)
        logger.info(f"Created OAuth client: {client_id}")
        return (client_id, client_secret)
    
    def get_client(self) -> Optional[OAuthClient]:
        """Get OAuth client configuration."""
        if not CLIENT_FILE.exists():
            return None
        
        try:
            with open(CLIENT_FILE, 'r') as f:
                data = json.load(f)
            return OAuthClient(**data)
        except Exception as e:
            logger.error(f"Error loading OAuth client: {e}")
            return None
    
    def verify_client(self, client_id: str, client_secret: Optional[str] = None) -> bool:
        """
        Verify OAuth client credentials.
        
        Args:
            client_id: Client ID to verify
            client_secret: Optional client secret for confidential clients
        
        Returns:
            True if valid, False otherwise
        """
        client = self.get_client()
        if not client or client.client_id != client_id:
            return False
        
        if client_secret is not None:
            return secrets.compare_digest(client.client_secret, client_secret)
        
        return True
    
    def create_authorization_code(
        self,
        client_id: str,
        redirect_uri: str,
        scope: Optional[str] = None,
        code_challenge: Optional[str] = None,
        code_challenge_method: Optional[str] = None
    ) -> str:
        """
        Create an authorization code.
        
        Args:
            client_id: OAuth client ID
            redirect_uri: Redirect URI for callback
            scope: Requested scope
            code_challenge: PKCE code challenge
            code_challenge_method: PKCE challenge method (S256 or plain)
        
        Returns:
            Authorization code
        """
        code = secrets.token_urlsafe(32)
        
        self.codes[code] = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method or "plain",
            "created_at": int(time.time()),
            "used": False
        }
        
        # Clean up old codes (> 10 minutes)
        current_time = int(time.time())
        expired_codes = [
            k for k, v in self.codes.items()
            if current_time - v["created_at"] > 600
        ]
        for k in expired_codes:
            del self.codes[k]
        
        return code
    
    def exchange_code_for_token(
        self,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_verifier: Optional[str] = None
    ) -> Optional[OAuthToken]:
        """
        Exchange authorization code for access token.
        
        Args:
            code: Authorization code
            client_id: OAuth client ID
            redirect_uri: Redirect URI (must match authorization request)
            code_verifier: PKCE code verifier
        
        Returns:
            Access token if valid, None otherwise
        """
        if code not in self.codes:
            logger.warning("Invalid authorization code")
            return None
        
        code_data = self.codes[code]
        
        # Verify code hasn't been used
        if code_data.get("used"):
            logger.warning("Authorization code already used")
            del self.codes[code]
            return None
        
        # Verify client_id and redirect_uri match
        if code_data["client_id"] != client_id:
            logger.warning("Client ID mismatch")
            return None
        
        # Allow redirect_uri to differ if both are HTTPS (ChatGPT uses different URIs)
        stored_redirect = code_data["redirect_uri"]
        if stored_redirect != redirect_uri:
            # Allow if both are HTTPS and same domain
            if not (stored_redirect.startswith("https://") and redirect_uri.startswith("https://")):
                logger.warning(
                    f"Redirect URI mismatch: stored={stored_redirect}, "
                    f"provided={redirect_uri}"
                )
                return None
        
        # Verify PKCE if used
        if code_data.get("code_challenge"):
            if not code_verifier:
                logger.warning("PKCE code verifier required but not provided")
                return None
            
            method = code_data.get("code_challenge_method", "plain")
            if method == "S256":
                # SHA-256 hash and base64url encode (no padding)
                import base64
                verifier_hash = hashlib.sha256(code_verifier.encode()).digest()
                verifier_challenge = base64.urlsafe_b64encode(verifier_hash).decode().rstrip('=')
            else:
                verifier_challenge = code_verifier
            
            if not secrets.compare_digest(verifier_challenge, code_data["code_challenge"]):
                logger.warning(
                    f"PKCE verification failed: "
                    f"expected={code_data['code_challenge']}, got={verifier_challenge}"
                )
                return None
        
        # Mark code as used
        code_data["used"] = True
        
        # Generate tokens
        access_token = secrets.token_urlsafe(48)
        refresh_token = secrets.token_urlsafe(48)
        
        token = OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=3600,
            refresh_token=refresh_token,
            scope=code_data.get("scope"),
            created_at=int(time.time())
        )
        
        # Store token
        self.tokens[access_token] = token
        self._save_tokens()
        
        logger.info(f"Issued access token for client {client_id}")
        return token
    
    def verify_token(self, access_token: str) -> bool:
        """
        Verify an access token.
        
        Args:
            access_token: Token to verify
        
        Returns:
            True if valid and not expired, False otherwise
        """
        if access_token not in self.tokens:
            return False
        
        token = self.tokens[access_token]
        
        # Check expiration
        if token.created_at:
            age = int(time.time()) - token.created_at
            if age > token.expires_in:
                logger.debug("Token expired")
                del self.tokens[access_token]
                self._save_tokens()
                return False
        
        return True
    
    def refresh_access_token(self, refresh_token: str) -> Optional[OAuthToken]:
        """
        Refresh an access token.
        
        Args:
            refresh_token: Refresh token
        
        Returns:
            New access token if valid, None otherwise
        """
        # Find token by refresh_token
        old_token = None
        for token in self.tokens.values():
            if token.refresh_token == refresh_token:
                old_token = token
                break
        
        if not old_token:
            logger.warning("Invalid refresh token")
            return None
        
        # Generate new access token
        access_token = secrets.token_urlsafe(48)
        
        token = OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=3600,
            refresh_token=refresh_token,  # Reuse refresh token
            scope=old_token.scope,
            created_at=int(time.time())
        )
        
        self.tokens[access_token] = token
        self._save_tokens()
        
        logger.info("Refreshed access token")
        return token
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke an access or refresh token.
        
        Args:
            token: Access token or refresh token to revoke
        
        Returns:
            True if revoked, False if not found
        """
        # Check if it's an access token
        if token in self.tokens:
            del self.tokens[token]
            self._save_tokens()
            return True
        
        # Check if it's a refresh token
        for access_token, token_data in list(self.tokens.items()):
            if token_data.refresh_token == token:
                del self.tokens[access_token]
                self._save_tokens()
                return True
        
        return False
    
    def create_state(self, redirect_uri: Optional[str] = None) -> str:
        """
        Create a state parameter for CSRF protection.
        
        Args:
            redirect_uri: Optional redirect URI to associate with state
        
        Returns:
            State parameter
        """
        state = secrets.token_urlsafe(32)
        
        self.states[state] = OAuthState(
            state=state,
            redirect_uri=redirect_uri,
            created_at=int(time.time())
        )
        
        # Clean up old states (> 10 minutes)
        current_time = int(time.time())
        expired_states = [
            k for k, v in self.states.items()
            if current_time - v.created_at > 600
        ]
        for k in expired_states:
            del self.states[k]
        
        return state
    
    def verify_state(self, state: str) -> bool:
        """
        Verify a state parameter.
        
        Args:
            state: State parameter to verify
        
        Returns:
            True if valid, False otherwise
        """
        if state not in self.states:
            return False
        
        # Check age
        state_data = self.states[state]
        age = int(time.time()) - state_data.created_at
        if age > 600:  # 10 minutes
            del self.states[state]
            return False
        
        return True
    
    def _load_tokens(self):
        """Load tokens from disk."""
        if not TOKENS_FILE.exists():
            return
        
        try:
            with open(TOKENS_FILE, 'r') as f:
                data = json.load(f)
            
            for access_token, token_data in data.items():
                self.tokens[access_token] = OAuthToken(**token_data)
            
            logger.info(f"Loaded {len(self.tokens)} OAuth tokens")
        except Exception as e:
            logger.error(f"Error loading tokens: {e}")
    
    def _save_tokens(self):
        """Save tokens to disk."""
        try:
            data = {
                access_token: asdict(token)
                for access_token, token in self.tokens.items()
            }
            
            with open(TOKENS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            
            TOKENS_FILE.chmod(0o600)
        except Exception as e:
            logger.error(f"Error saving tokens: {e}")


# Global OAuth manager instance
_oauth_manager: Optional[OAuthManager] = None


def get_oauth_manager() -> OAuthManager:
    """Get or create the global OAuth manager instance."""
    global _oauth_manager
    if _oauth_manager is None:
        _oauth_manager = OAuthManager()
    return _oauth_manager
