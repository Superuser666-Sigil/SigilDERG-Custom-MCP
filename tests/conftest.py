# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Pytest configuration and shared fixtures for Sigil MCP Server tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from sigil_mcp.indexer import SigilIndex
from sigil_mcp.auth import API_KEY_FILE
from sigil_mcp.oauth import CLIENT_FILE, TOKENS_FILE
import sigil_mcp.config as sigil_config


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
    """Create a dummy embedding function for testing."""
    def embed_fn(texts):
        """Generate deterministic embeddings for testing."""
        dim = 768
        embeddings = np.random.randn(len(texts), dim).astype('float32')
        # Make deterministic based on text content
        for i, text in enumerate(texts):
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            embeddings[i] = np.random.randn(dim).astype('float32')
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        return embeddings
    return embed_fn


@pytest.fixture
def test_index(test_index_path, dummy_embed_fn):
    """Create a SigilIndex instance for testing."""
    original_config = sigil_config._config
    cfg = sigil_config.Config()
    cfg.config_data.setdefault("embeddings", {})["enabled"] = True
    sigil_config._config = cfg

    index = SigilIndex(
        index_path=test_index_path,
        embed_fn=dummy_embed_fn,
        embed_model="test-model"
    )
    yield index
    # Cleanup
    index.repos_db.close()
    index.trigrams_db.close()
    sigil_config._config = original_config


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
def embeddings_enabled_index(temp_dir, test_repo_path, dummy_embed_fn):
    """Create an index with embeddings enabled for LanceDB-covered tests."""

    # Temporarily enable embeddings in global config for this test run
    original_config = sigil_config._config
    cfg = sigil_config.Config()
    cfg.config_data.setdefault("embeddings", {})["enabled"] = True
    sigil_config._config = cfg

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
        index.repos_db.close()
        index.trigrams_db.close()
        sigil_config._config = original_config


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
    # Backup existing API key if present
    backup = None
    if API_KEY_FILE.exists():
        backup = API_KEY_FILE.read_text()
        API_KEY_FILE.unlink()
    
    yield
    
    # Restore or cleanup
    if backup:
        API_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        API_KEY_FILE.write_text(backup)
    elif API_KEY_FILE.exists():
        API_KEY_FILE.unlink()


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
