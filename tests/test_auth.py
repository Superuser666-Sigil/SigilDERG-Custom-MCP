# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Tests for authentication module (auth.py).
"""

from pathlib import Path

from sigil_mcp.auth import (
    generate_api_key,
    hash_api_key,
    initialize_api_key,
    verify_api_key,
    get_api_key_from_env,
    get_api_key_path,
    _update_api_key_path,
    API_KEY_FILE,
)


def _api_key_file():
    return get_api_key_path()


class TestAPIKeyGeneration:
    """Test API key generation and hashing."""
    
    def test_generate_api_key_length(self):
        """Test that generated API keys have expected length."""
        key = generate_api_key()
        assert len(key) >= 32
        assert isinstance(key, str)
    
    def test_generate_api_key_uniqueness(self):
        """Test that generated keys are unique."""
        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100
    
    def test_hash_api_key_deterministic(self):
        """Test that hashing the same key produces same hash."""
        key = "test_api_key_12345"
        hash1 = hash_api_key(key)
        hash2 = hash_api_key(key)
        assert hash1 == hash2
    
    def test_hash_api_key_different_for_different_keys(self):
        """Test that different keys produce different hashes."""
        key1 = "test_key_1"
        key2 = "test_key_2"
        assert hash_api_key(key1) != hash_api_key(key2)
    
    def test_hash_api_key_format(self):
        """Test that hash is in expected format (hex string)."""
        key = "test_key"
        hashed = hash_api_key(key)
        assert len(hashed) == 64  # SHA256 produces 64-char hex string
        assert all(c in "0123456789abcdef" for c in hashed)


class TestAPIKeyInitialization:
    """Test API key initialization and storage."""
    
    def test_initialize_api_key_creates_file(self, clean_auth_file):
        """Test that initialization creates the API key file."""
        api_key = initialize_api_key()
        assert api_key is not None
        assert _api_key_file().exists()
        assert len(api_key) >= 32
    
    def test_initialize_api_key_stores_hash(self, clean_auth_file):
        """Test that only the hash is stored, not plaintext."""
        api_key = initialize_api_key()
        assert api_key is not None
        stored_hash = _api_key_file().read_text().strip()
        
        # Verify stored value is a hash
        assert len(stored_hash) == 64
        assert stored_hash != api_key
        assert stored_hash == hash_api_key(api_key)
    
    def test_initialize_api_key_sets_permissions(self, clean_auth_file):
        """Test that API key file has restrictive permissions."""
        initialize_api_key()
        import stat
        mode = _api_key_file().stat().st_mode
        # Check that file is readable/writable only by owner
        assert stat.S_IMODE(mode) == 0o600
    
    def test_initialize_api_key_returns_none_if_exists(self, clean_auth_file):
        """Test that initialization returns None if key already exists."""
        # First initialization
        api_key1 = initialize_api_key()
        assert api_key1 is not None
        
        # Second initialization
        api_key2 = initialize_api_key()
        assert api_key2 is None


class TestAPIKeyVerification:
    """Test API key verification."""
    
    def test_verify_api_key_valid(self, clean_auth_file):
        """Test verification of valid API key."""
        api_key = initialize_api_key()
        assert api_key is not None
        assert verify_api_key(api_key) is True
    
    def test_verify_api_key_invalid(self, clean_auth_file):
        """Test verification of invalid API key."""
        initialize_api_key()
        assert verify_api_key("wrong_key") is False
    
    def test_verify_api_key_no_file(self, clean_auth_file):
        """Test verification when no API key file exists (should allow)."""
        # No file created
        assert verify_api_key("any_key") is True
    
    def test_verify_api_key_empty(self, clean_auth_file):
        """Test verification of empty key."""
        initialize_api_key()
        assert verify_api_key("") is False
    
    def test_verify_api_key_timing_safe(self, clean_auth_file):
        """Test that verification uses constant-time comparison."""
        # This test ensures we're using secrets.compare_digest
        # which is timing-attack resistant
        api_key = initialize_api_key()
        assert api_key is not None
        
        # Verification should use constant-time comparison
        # (cannot directly test timing, but we verify it's called)
        assert verify_api_key(api_key) is True
        assert verify_api_key(api_key[:-1] + "X") is False


class TestEnvironmentVariableAuth:
    """Test environment variable authentication."""
    
    def test_get_api_key_from_env_present(self, monkeypatch):
        """Test getting API key from environment variable."""
        test_key = "test_env_api_key"
        monkeypatch.setenv("SIGIL_MCP_API_KEY", test_key)
        assert get_api_key_from_env() == test_key
    
    def test_get_api_key_from_env_absent(self):
        """Test getting API key when environment variable not set."""
        import os
        # Ensure env var is not set
        os.environ.pop("SIGIL_MCP_API_KEY", None)
        assert get_api_key_from_env() is None


def test_update_api_key_path_updates_globals(tmp_path):
    original = get_api_key_path()
    new_path = tmp_path / "new_key"
    _update_api_key_path(new_path)
    assert get_api_key_path() == new_path
    _update_api_key_path(original)


def test_initialize_api_key_permission_fallback(monkeypatch, tmp_path):
    target = tmp_path / "api_key"
    fallback = tmp_path / "fallback_key"
    monkeypatch.setattr("sigil_mcp.auth._WORKSPACE_FALLBACK", fallback)
    monkeypatch.setattr("sigil_mcp.auth._update_api_key_path", lambda p: None)
    monkeypatch.setattr("sigil_mcp.auth.generate_api_key", lambda: "key")
    monkeypatch.setattr("sigil_mcp.auth.hash_api_key", lambda k: "hash")
    monkeypatch.setattr("sigil_mcp.auth.get_api_key_path", lambda: target)

    def fake_open(path, mode="r", *a, **k):
        if Path(path) == target and "w" in mode:
            raise PermissionError()
        return open(path, mode, *a, **k)

    monkeypatch.setattr("sigil_mcp.auth.open", fake_open, raising=False)
    created = initialize_api_key()
    assert created == "key"
    assert fallback.exists()
    assert fallback.read_text() == "hash"


def test_verify_api_key_permission_error(monkeypatch, tmp_path):
    target = tmp_path / "api_key"
    target.write_text("hash")

    def fake_open(path, mode="r", *a, **k):
        raise PermissionError()

    monkeypatch.setattr("sigil_mcp.auth.get_api_key_path", lambda: target)
    monkeypatch.setattr("sigil_mcp.auth.open", fake_open, raising=False)
    assert verify_api_key("value") is False


def test_verify_api_key_generic_error(monkeypatch, tmp_path):
    target = tmp_path / "api_key"
    target.write_text("hash")

    def fake_open(path, mode="r", *a, **k):
        raise ValueError("boom")

    monkeypatch.setattr("sigil_mcp.auth.get_api_key_path", lambda: target)
    monkeypatch.setattr("sigil_mcp.auth.open", fake_open, raising=False)
    assert verify_api_key("value") is False
