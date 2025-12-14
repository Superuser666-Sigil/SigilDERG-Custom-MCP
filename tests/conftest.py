# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Pytest configuration and shared fixtures for Sigil MCP Server tests.
"""

# ruff: noqa: E402
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Use the in-memory LanceDB stub during tests to avoid long-running native setup
os.environ.setdefault("SIGIL_MCP_LANCEDB_STUB", "1")
# Route OAuth storage to a writable temp directory for tests
TEST_OAUTH_DIR = Path(tempfile.mkdtemp(prefix="sigil_oauth_"))
os.environ.setdefault("SIGIL_MCP_OAUTH_DIR", str(TEST_OAUTH_DIR))

import sigil_mcp.config as sigil_config
from sigil_mcp.indexer import SigilIndex
from sigil_mcp.oauth import CLIENT_FILE, TOKENS_FILE


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def test_index_path(temp_dir):
    """Create a temporary index directory."""
    index_path = temp_dir / ".test_index"
    index_path.mkdir(parents=True, exist_ok=True)
    yield index_path


@pytest.fixture
def test_repo_path(temp_dir):
    """Create a temporary repository with sample files."""
    repo_path = temp_dir / "test_repo"
    repo_path.mkdir(parents=True, exist_ok=True)

    # Create sample Python files
    (repo_path / "main.py").write_text("""
def hello_world():
    '''Say hello to the world.'''
    print("Hello, World!")

class Calculator:
    '''Simple calculator class.'''

    def add(self, a, b):
        '''Add two numbers.'''
        return a + b

    def subtract(self, a, b):
        '''Subtract b from a.'''
        return a - b

if __name__ == "__main__":
    hello_world()
""")

    (repo_path / "utils.py").write_text("""
def process_data(data):
    '''Process input data.'''
    result = []
    for item in data:
        result.append(item.strip())
    return result

def validate_input(value):
    '''Validate user input.'''
    if not value:
        raise ValueError("Value cannot be empty")
    return True
""")

    # Create a subdirectory with more files
    subdir = repo_path / "lib"
    subdir.mkdir()

    (subdir / "helper.py").write_text("""
def format_output(text):
    '''Format text for output.'''
    return text.upper()

class Logger:
    '''Simple logging class.'''

    def __init__(self, name):
        self.name = name

    def log(self, message):
        '''Log a message.'''
        print(f"[{self.name}] {message}")
""")

    yield repo_path


@pytest.fixture
def dummy_embed_fn():
    """Create a deterministic embedding function for testing."""
    import hashlib

    from numpy.random import default_rng

    def embed_fn(texts):
        dim = 768
        embeddings = np.empty((len(texts), dim), dtype="float32")

        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # Use int from digest to seed a local RNG; avoid global np.random state
            seed_int = int.from_bytes(digest[:8], "big", signed=False)
            rng = default_rng(seed_int)
            embeddings[i] = rng.standard_normal(dim).astype("float32")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        return embeddings

    return embed_fn


@pytest.fixture
def test_index(test_index_path, dummy_embed_fn, monkeypatch):
    """Create a SigilIndex instance for testing."""
    cfg = sigil_config.Config()
    cfg.config_data["embeddings"] = {"enabled": True}
    cfg.config_data["index"] = {"ignore_patterns": []}
    monkeypatch.setattr(sigil_config, "_config", cfg)

    index = SigilIndex(
        index_path=test_index_path,
        embed_fn=dummy_embed_fn,
        embed_model="test-model"
    )
    yield index
    # Cleanup
    index.close()


@pytest.fixture
def indexed_repo(test_index, test_repo_path):
    """Create an indexed repository for testing."""
    stats = test_index.index_repository("test_repo", test_repo_path, force=True)
    return {
        "index": test_index,
        "repo_path": test_repo_path,
        "repo_name": "test_repo",
        "stats": stats
    }


@pytest.fixture
def embeddings_enabled_index(temp_dir, test_repo_path, dummy_embed_fn, monkeypatch):
    """Create an index with embeddings enabled for LanceDB-covered tests."""

    # Temporarily enable embeddings in global config for this test run
    cfg = sigil_config.Config()
    cfg.config_data.setdefault("embeddings", {})["enabled"] = True
    monkeypatch.setattr(sigil_config, "_config", cfg)

    index_path = temp_dir / ".test_index_vectors"
    index_path.mkdir(parents=True, exist_ok=True)

    index = SigilIndex(
        index_path=index_path,
        embed_fn=dummy_embed_fn,
        embed_model="test-model",
    )

    try:
        yield {
            "index": index,
            "repo_path": test_repo_path,
            "repo_name": "test_repo",
        }
    finally:
        index.close()


@pytest.fixture
def test_config_file(temp_dir):
    """Create a temporary config file for testing."""
    config_path = temp_dir / "config.json"
    config_data = {
        "server": {
            "name": "test_server",
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "DEBUG"
        },
        "authentication": {
            "enabled": True,
            "oauth_enabled": True,
            "allow_local_bypass": True,
            "allowed_ips": ["192.168.1.1"]
        },
        "repositories": {
            "test_repo": "/path/to/test/repo"
        },
        "index": {
            "path": str(temp_dir / ".test_index")
        }
    }

    import json
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    yield config_path


@pytest.fixture
def clean_auth_file():
    """Ensure clean auth state before and after tests."""
    from sigil_mcp.auth import get_api_key_path

    path = get_api_key_path()
    backup = None
    if path.exists():
        backup = path.read_text()
        path.unlink()

    yield

    if backup:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(backup)
    elif path.exists():
        path.unlink()


@pytest.fixture
def clean_oauth_files():
    """Ensure clean OAuth state before and after tests."""
    # Backup existing OAuth files
    backups = {}
    for file in [CLIENT_FILE, TOKENS_FILE]:
        if file.exists():
            backups[file] = file.read_text()
            file.unlink()

    yield

    # Restore or cleanup
    for file, content in backups.items():
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)

    # Cleanup test files
    for file in [CLIENT_FILE, TOKENS_FILE]:
        if file.exists() and file not in backups:
            file.unlink()
