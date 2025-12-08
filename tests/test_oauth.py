# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Tests for OAuth2 authentication module (oauth.py).
"""

import time
import json
from sigil_mcp.oauth import (
    OAuthManager,
    OAuthClient,
    OAuthToken,
    get_oauth_manager,
    CLIENT_FILE,
    TOKENS_FILE
)


class TestOAuthClientManagement:
    """Test OAuth client initialization and management."""
    
    def test_initialize_client_creates_credentials(self, clean_oauth_files):
        """Test that client initialization creates credentials."""
        manager = OAuthManager()
        result = manager.initialize_client()
        
        assert result is not None
        client_id, client_secret = result
        assert client_id.startswith("sigil_")
        assert len(client_secret) >= 32
        assert CLIENT_FILE.exists()
    
    def test_initialize_client_returns_none_if_exists(self, clean_oauth_files):
        """Test that initialization returns None if client exists."""
        manager = OAuthManager()
        
        # First initialization
        result1 = manager.initialize_client()
        assert result1 is not None
        
        # Second initialization
        result2 = manager.initialize_client()
        assert result2 is None
    
    def test_get_client_returns_oauth_client(self, clean_oauth_files):
        """Test getting OAuth client configuration."""
        manager = OAuthManager()
        manager.initialize_client()
        
        client = manager.get_client()
        assert client is not None
        assert isinstance(client, OAuthClient)
        assert client.client_id.startswith("sigil_")
        assert len(client.redirect_uris) > 0
    
    def test_verify_client_valid(self, clean_oauth_files):
        """Test verifying valid client credentials."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, client_secret = result
        
        # Verify with correct credentials
        assert manager.verify_client(client_id, client_secret) is True
    
    def test_verify_client_invalid_id(self, clean_oauth_files):
        """Test verifying invalid client ID."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, client_secret = result
        
        assert manager.verify_client("wrong_id", client_secret) is False
    
    def test_verify_client_invalid_secret(self, clean_oauth_files):
        """Test verifying invalid client secret."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        
        assert manager.verify_client(client_id, "wrong_secret") is False
    
    def test_verify_client_without_secret(self, clean_oauth_files):
        """Test verifying client ID only (public client)."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        
        # Public clients don't provide secret
        assert manager.verify_client(client_id) is True


class TestAuthorizationCodeFlow:
    """Test OAuth2 authorization code flow."""
    
    def test_create_authorization_code(self, clean_oauth_files):
        """Test creating authorization code."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        assert len(code) >= 32
        assert code in manager.codes
    
    def test_authorization_code_has_metadata(self, clean_oauth_files):
        """Test that authorization code stores necessary metadata."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        
        redirect_uri = "http://localhost:8080/callback"
        scope = "read write"
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri=redirect_uri,
            scope=scope
        )
        
        code_data = manager.codes[code]
        assert code_data["client_id"] == client_id
        assert code_data["redirect_uri"] == redirect_uri
        assert code_data["scope"] == scope
        assert code_data["used"] is False
    
    def test_exchange_code_for_token_success(self, clean_oauth_files):
        """Test successfully exchanging code for token."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        redirect_uri = "http://localhost:8080/callback"
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri=redirect_uri
        )
        
        token = manager.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri
        )
        
        assert token is not None
        assert isinstance(token, OAuthToken)
        assert len(token.access_token) >= 32
        assert token.refresh_token is not None
    
    def test_exchange_code_invalid_code(self, clean_oauth_files):
        """Test exchanging invalid authorization code."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        
        token = manager.exchange_code_for_token(
            code="invalid_code",
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        assert token is None
    
    def test_exchange_code_already_used(self, clean_oauth_files):
        """Test that authorization code can only be used once."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        redirect_uri = "http://localhost:8080/callback"
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri=redirect_uri
        )
        
        # First exchange succeeds
        token1 = manager.exchange_code_for_token(code, client_id, redirect_uri)
        assert token1 is not None
        
        # Second exchange fails
        token2 = manager.exchange_code_for_token(code, client_id, redirect_uri)
        assert token2 is None
    
    def test_exchange_code_client_mismatch(self, clean_oauth_files):
        """Test that client_id must match."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        redirect_uri = "http://localhost:8080/callback"
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri=redirect_uri
        )
        
        token = manager.exchange_code_for_token(
            code=code,
            client_id="wrong_client",
            redirect_uri=redirect_uri
        )
        
        assert token is None


class TestPKCEFlow:
    """Test PKCE (Proof Key for Code Exchange) flow."""
    
    def test_pkce_s256_verification_success(self, clean_oauth_files):
        """Test successful PKCE S256 verification."""
        import base64
        import hashlib
        
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        redirect_uri = "http://localhost:8080/callback"
        
        # Generate code verifier and challenge
        code_verifier = "test_verifier_1234567890abcdefghijklmnopqrstuvwxyz"
        verifier_hash = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(verifier_hash).decode().rstrip('=')
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method="S256"
        )
        
        token = manager.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier
        )
        
        assert token is not None
    
    def test_pkce_s256_verification_failure(self, clean_oauth_files):
        """Test failed PKCE S256 verification with wrong verifier."""
        import base64
        import hashlib
        
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        redirect_uri = "http://localhost:8080/callback"
        
        code_verifier = "test_verifier_123"
        verifier_hash = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(verifier_hash).decode().rstrip('=')
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method="S256"
        )
        
        # Use wrong verifier
        token = manager.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_verifier="wrong_verifier"
        )
        
        assert token is None
    
    def test_pkce_plain_verification(self, clean_oauth_files):
        """Test PKCE plain method verification."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        redirect_uri = "http://localhost:8080/callback"
        
        code_verifier = "test_verifier_plain"
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_verifier,
            code_challenge_method="plain"
        )
        
        token = manager.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier
        )
        
        assert token is not None


class TestTokenManagement:
    """Test OAuth token management."""
    
    def test_verify_token_valid(self, clean_oauth_files):
        """Test verifying valid access token."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        token = manager.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        assert token is not None
        assert manager.verify_token(token.access_token) is True
    
    def test_verify_token_invalid(self, clean_oauth_files):
        """Test verifying invalid access token."""
        manager = OAuthManager()
        assert manager.verify_token("invalid_token") is False
    
    def test_verify_token_expired(self, clean_oauth_files):
        """Test that expired tokens are rejected."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        token = manager.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        assert token is not None
        # Manually expire token
        token.created_at = int(time.time()) - 7200  # 2 hours ago
        manager.tokens[token.access_token] = token
        
        assert manager.verify_token(token.access_token) is False
    
    def test_refresh_access_token_success(self, clean_oauth_files):
        """Test successfully refreshing access token."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        old_token = manager.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        assert old_token is not None
        assert old_token.refresh_token is not None
        new_token = manager.refresh_access_token(old_token.refresh_token)
        
        assert new_token is not None
        assert new_token.access_token != old_token.access_token
        assert new_token.refresh_token == old_token.refresh_token
    
    def test_refresh_access_token_invalid(self, clean_oauth_files):
        """Test refreshing with invalid refresh token."""
        manager = OAuthManager()
        new_token = manager.refresh_access_token("invalid_refresh_token")
        assert new_token is None
    
    def test_revoke_access_token(self, clean_oauth_files):
        """Test revoking access token."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        token = manager.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        assert token is not None
        # Revoke token
        assert manager.revoke_token(token.access_token) is True
        
        # Verify token is no longer valid
        assert manager.verify_token(token.access_token) is False
    
    def test_revoke_refresh_token(self, clean_oauth_files):
        """Test revoking refresh token."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        token = manager.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        assert token is not None
        assert token.refresh_token is not None
        # Revoke refresh token
        assert manager.revoke_token(token.refresh_token) is True
        
        # Access token should also be revoked
        assert manager.verify_token(token.access_token) is False


class TestStateManagement:
    """Test OAuth state parameter management (CSRF protection)."""
    
    def test_create_state(self, clean_oauth_files):
        """Test creating state parameter."""
        manager = OAuthManager()
        state = manager.create_state()
        
        assert len(state) >= 32
        assert state in manager.states
    
    def test_verify_state_valid(self, clean_oauth_files):
        """Test verifying valid state parameter."""
        manager = OAuthManager()
        state = manager.create_state()
        assert manager.verify_state(state) is True
    
    def test_verify_state_invalid(self, clean_oauth_files):
        """Test verifying invalid state parameter."""
        manager = OAuthManager()
        assert manager.verify_state("invalid_state") is False
    
    def test_verify_state_expired(self, clean_oauth_files):
        """Test that expired state is rejected."""
        manager = OAuthManager()
        state = manager.create_state()
        
        # Manually expire state
        state_data = manager.states[state]
        state_data.created_at = int(time.time()) - 700  # 11+ minutes ago
        
        assert manager.verify_state(state) is False
    
    def test_state_cleanup(self, clean_oauth_files):
        """Test that old states are cleaned up."""
        manager = OAuthManager()
        
        # Create state and manually age it
        state1 = manager.create_state()
        manager.states[state1].created_at = int(time.time()) - 700
        
        # Create new state (should trigger cleanup)
        state2 = manager.create_state()
        
        # Old state should be removed
        assert state1 not in manager.states
        assert state2 in manager.states


class TestTokenPersistence:
    """Test token persistence to disk."""
    
    def test_tokens_saved_to_disk(self, clean_oauth_files):
        """Test that tokens are saved to disk."""
        manager = OAuthManager()
        result = manager.initialize_client()
        assert result is not None
        client_id, _ = result
        
        code = manager.create_authorization_code(
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        token = manager.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        assert token is not None
        assert TOKENS_FILE.exists()
        
        # Verify token is in file
        with open(TOKENS_FILE, 'r') as f:
            saved_tokens = json.load(f)
        
        assert token.access_token in saved_tokens
    
    def test_tokens_loaded_from_disk(self, clean_oauth_files):
        """Test that tokens are loaded from disk on initialization."""
        # First manager creates token
        manager1 = OAuthManager()
        result = manager1.initialize_client()
        assert result is not None
        client_id, _ = result
        
        code = manager1.create_authorization_code(
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        token = manager1.exchange_code_for_token(
            code=code,
            client_id=client_id,
            redirect_uri="http://localhost:8080/callback"
        )
        
        assert token is not None
        # Second manager loads tokens
        manager2 = OAuthManager()
        
        assert manager2.verify_token(token.access_token) is True


class TestGlobalOAuthManager:
    """Test global OAuth manager instance."""
    
    def test_get_oauth_manager_singleton(self):
        """Test that get_oauth_manager returns singleton."""
        manager1 = get_oauth_manager()
        manager2 = get_oauth_manager()
        assert manager1 is manager2
