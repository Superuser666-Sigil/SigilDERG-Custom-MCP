<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Sigil MCP Server Operations Runbook

**Version:** 1.1  
**Last Updated:** 2025-12-03  
**Recommended Server Version:** v0.3.1 or later

This runbook provides operational procedures for running, troubleshooting, and maintaining the Sigil MCP Server in production and development environments.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation Procedures](#installation-procedures)
3. [Configuration Management](#configuration-management)
4. [Server Operations](#server-operations)
5. [Index Management](#index-management)
6. [Authentication & Security](#authentication--security)
7. [File Watching](#file-watching)
8. [Direct Python Testing](#direct-python-testing)
9. [Troubleshooting](#troubleshooting)
10. [Monitoring & Health Checks](#monitoring--health-checks)
11. [Backup & Recovery](#backup--recovery)
12. [Performance Tuning](#performance-tuning)
13. [Common Tasks](#common-tasks)

---

## Quick Start

### Minimal Setup (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP.git
cd SigilDERG-Custom-MCP

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure repositories
cp config.example.json config.json
# Edit config.json with your repository paths

# 4. Start server
python -m sigil_mcp.server

# 5. Note OAuth credentials from startup output
```

### Production Setup (15 minutes)

See [Installation Procedures](#installation-procedures) section.

---

## Installation Procedures

### Base Installation

#### Prerequisites

- Python 3.12 or higher
- pip package manager
- Universal Ctags (optional, for symbol extraction)

#### Step 1: Install Python Package

```bash
# Option A: Install from source
git clone https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP.git
cd SigilDERG-Custom-MCP
pip install -e .

# Option B: Install from PyPI (when available)
pip install sigil-mcp-server
```

#### Step 2: Install Optional Dependencies

```bash
# For file watching
pip install sigil-mcp-server[watch]

# For vector embeddings
pip install sigil-mcp-server[embeddings]

# For development
pip install sigil-mcp-server[dev]

# All optional features
pip install sigil-mcp-server[watch,embeddings,dev]
```

#### Step 3: Install Universal Ctags

**macOS:**
```bash
brew install universal-ctags
```

**Ubuntu/Debian:**
```bash
sudo apt install universal-ctags
```

**Arch Linux:**
```bash
sudo pacman -S ctags
```

**Verify installation:**
```bash
ctags --version
# Should show "Universal Ctags"
```

### Configuration Setup

#### Create Configuration File

```bash
cp config.example.json config.json
```

#### Edit Configuration

Minimal `config.json`:
```json
{
  "repositories": {
    "project1": "/absolute/path/to/project1",
    "project2": "/absolute/path/to/project2"
  }
}
```

Full example: See [config.example.json](../config.example.json)

#### Environment Variables (Alternative)

```bash
export SIGIL_REPO_MAP="proj1:/path/to/proj1;proj2:/path/to/proj2"
export SIGIL_MCP_HOST=0.0.0.0
export SIGIL_MCP_PORT=8000
export SIGIL_INDEX_PATH=~/.sigil_index
```

### First Run

```bash
# Start server
python -m sigil_mcp.server

# Expected output:
# OAuth Client ID: abc123...
# OAuth Client Secret: xyz789...
# Server running on http://127.0.0.1:8000
```

**⚠️ Important:** Save OAuth credentials securely!

---

## Configuration Management

### Configuration File Location

Default search order:
1. `./config.json` (current directory)
2. `~/.config/sigil-mcp/config.json`
3. Environment variables

### Configuration Schema

```json
{
  "server": {
    "name": "sigil_repos",
    "host": "127.0.0.1",
    "port": 8000,
    "log_level": "INFO"
  },
  "authentication": {
    "enabled": true,
    "oauth_enabled": true,
    "allow_local_bypass": true,
    "allowed_ips": []
  },
  "repositories": {
    "repo_name": "/absolute/path"
  },
  "index": {
    "path": "~/.sigil_index",
    "skip_dirs": ["node_modules", "venv", ".git"],
    "skip_files": ["*.pyc", "*.so", "*.pdf"]
  },
  "watch": {
    "enabled": true,
    "debounce_seconds": 2.0,
    "ignore_dirs": [".git", "__pycache__"],
    "ignore_extensions": [".pyc", ".so"]
  },
  "embeddings": {
    "enabled": false,
    "provider": "sentence-transformers",
    "model": "all-MiniLM-L6-v2",
    "dimension": 384,
    "cache_dir": "~/.cache/sigil-embeddings"
  }
}
```

### Validating Configuration

```bash
# Test configuration loading
python -c "from sigil_mcp.config import Config; from pathlib import Path; \
  cfg = Config(Path('config.json')); \
  print(f'Loaded {len(cfg.repositories)} repositories')"
```

### Changing Configuration

1. Edit `config.json`
2. Restart server for changes to take effect
3. Most settings require full restart (no hot-reload)

---

## Server Operations

### Starting the Server

#### Foreground (Development)

```bash
python -m sigil_mcp.server
```

#### Background (Production)

```bash
# Using nohup
nohup python -m sigil_mcp.server > sigil.log 2>&1 &
echo $! > sigil.pid

# Using systemd (recommended)
sudo systemctl start sigil-mcp
```

#### Docker (Future)

```bash
docker run -d \
  -v /path/to/repos:/repos \
  -v ~/.sigil_index:/index \
  -p 8000:8000 \
  sigil-mcp-server
```

### Stopping the Server

```bash
# Find process
ps aux | grep sigil_mcp

# Graceful shutdown
kill -TERM $(cat sigil.pid)

# Force kill (last resort)
kill -9 $(cat sigil.pid)

# systemd
sudo systemctl stop sigil-mcp
```

### Restarting the Server

```bash
# Manual
kill -TERM $(cat sigil.pid) && sleep 2 && python -m sigil_mcp.server &

# systemd
sudo systemctl restart sigil-mcp
```

### Checking Server Status

```bash
# Check if running
curl http://localhost:8000/health

# Expected: {"status": "ok"}

# Check process
ps aux | grep sigil_mcp

# Check logs
tail -f sigil.log
```

### Exposing Server Externally

#### Using ngrok (Development)

```bash
ngrok http 8000
# Note the https URL (e.g., https://abc123.ngrok.io)
```

**ChatGPT Compatibility**: The server is configured with:
- `streamable_http_path="/"` - MCP endpoint at root path
- `enable_dns_rebinding_protection=False` - Accepts ngrok Host headers
- `json_response=True` - Returns JSON instead of SSE streams

These settings are required for ChatGPT's MCP connector which:
- Sends non-standard `Content-Type: application/octet-stream`
- Sends ngrok domains in Host headers
- Expects MCP endpoint at `/` not `/mcp`

#### Using Reverse Proxy (Production)

**nginx example:**
```nginx
server {
    listen 443 ssl;
    server_name sigil.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Index Management

### Building Initial Index

```bash
# Using MCP tool (from ChatGPT)
"Index the my_project repository"

# Programmatically
python -c "
from sigil_mcp.indexer import SigilIndex
from pathlib import Path

index = SigilIndex(Path('~/.sigil_index').expanduser())
stats = index.index_repository('my_project', Path('/path/to/project'))
print(f'Indexed {stats[\"files\"]} files, {stats[\"symbols\"]} symbols')
"
```

### Rebuilding Index (Force)

```bash
# Via MCP tool
"Force re-index my_project repository"

# Programmatically with force=True
python -c "
from sigil_mcp.indexer import SigilIndex
from pathlib import Path

index = SigilIndex(Path('~/.sigil_index').expanduser())
stats = index.index_repository('my_project', Path('/path/to/project'), force=True)
print(f'Re-indexed {stats[\"files\"]} files')
"
```

### Viewing Index Statistics

```bash
# Via MCP tool
"Show index statistics"

# Direct query
python -c "
from sigil_mcp.indexer import SigilIndex
from pathlib import Path

index = SigilIndex(Path('~/.sigil_index').expanduser())
stats = index.get_stats()
print(f'Repositories: {stats[\"repos\"]}')
print(f'Files: {stats[\"files\"]}')
print(f'Symbols: {stats[\"symbols\"]}')
print(f'Trigrams: {stats[\"trigrams\"]}')
"
```

### Index Location

Default: `~/.sigil_index/`

Contents:
- `repos.db` - Repository and document metadata, symbols
- `trigrams.db` - Trigram index for fast text search
- `blobs/` - Compressed file contents
- `vectors/` - Vector embeddings (if enabled)

### Index Cleanup

```bash
# Remove index directory (requires rebuild)
rm -rf ~/.sigil_index

# Or specify in config
rm -rf /path/to/custom/index
```

### Index Backup

```bash
# Backup entire index
tar -czf sigil-index-backup-$(date +%Y%m%d).tar.gz ~/.sigil_index

# Backup specific repository
sqlite3 ~/.sigil_index/repos.db ".backup repos-backup.db"
```

---

## Authentication & Security

### OAuth Setup

#### First Run - Generate Credentials

```bash
python -m sigil_mcp.server
# Outputs:
# OAuth Client ID: <save this>
# OAuth Client Secret: <save this>
```

#### Viewing Existing Credentials

```bash
python -m sigil_mcp.manage_auth show-oauth
```

#### Regenerating Credentials

```bash
# ⚠️ This invalidates all existing tokens
python -m sigil_mcp.manage_auth regenerate-oauth
```

### API Key Management

#### Creating API Keys

```bash
# Interactive
python -m sigil_mcp.manage_auth create-key

# Programmatic
python -m sigil_mcp.manage_auth create-key --name "ChatGPT" --expires 365
```

#### Listing API Keys

```bash
python -m sigil_mcp.manage_auth list-keys
```

#### Revoking API Keys

```bash
python -m sigil_mcp.manage_auth revoke-key <key_id>
```

### Security Best Practices

1. **Never commit credentials** to version control
2. **Use HTTPS** in production (ngrok or reverse proxy)
3. **Rotate OAuth secrets** periodically
4. **Limit API key expiration** to reasonable periods
5. **Enable IP whitelisting** for known clients
6. **Use local bypass only** for development

### IP Whitelisting

In `config.json`:
```json
{
  "authentication": {
    "allowed_ips": ["192.168.1.100", "10.0.0.0/8"]
  }
}
```

### Disabling Authentication (Development Only)

```json
{
  "authentication": {
    "enabled": false
  }
}
```

**⚠️ Never disable auth in production!**

---

## File Watching

### Enabling File Watching

```bash
# 1. Install watchdog
pip install sigil-mcp-server[watch]

# 2. Enable in config
{
  "watch": {
    "enabled": true
  }
}

# 3. Restart server
```

### Configuring Watch Behavior

```json
{
  "watch": {
    "enabled": true,
    "debounce_seconds": 2.0,
    "ignore_dirs": [
      ".git",
      "__pycache__",
      "node_modules",
      "coverage"
    ],
    "ignore_extensions": [
      ".pyc",
      ".so",
      ".pdf"
    ]
  }
}
```

### Verifying File Watching

```bash
# Check logs for watcher initialization
tail -f sigil.log | grep -i watch

# Expected:
# File watcher initialized
# Watching repository: my_project at /path/to/project
```

### Testing File Watching

```bash
# 1. Start server with watching enabled
python -m sigil_mcp.server

# 2. In another terminal, modify a file
echo "# test" >> /path/to/project/test.py

# 3. Check logs
# Expected:
# File modified: test.py in my_project
# Re-indexed test.py after modified
```

### Disabling File Watching

```bash
# Temporarily
export SIGIL_MCP_WATCH_ENABLED=false
python -m sigil_mcp.server

# Or in config.json
{
  "watch": {
    "enabled": false
  }
}
```

### Troubleshooting File Watching

**Symptom:** "watchdog not available" message

**Solution:**
```bash
pip install watchdog>=3.0.0
```

**Symptom:** Excessive CPU usage

**Solution:** Increase debounce time or add more ignore patterns
```json
{
  "watch": {
    "debounce_seconds": 5.0
  }
}
```

---

## Troubleshooting

### Common Issues

#### Server Won't Start

**Symptom:** `Address already in use`

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000
kill -9 <PID>

# Or change port
export SIGIL_MCP_PORT=8001
```

#### Repository Not Found

**Symptom:** `Repository 'name' not found`

**Solution:**
```bash
# Check repository configuration
python -c "from sigil_mcp.config import Config; from pathlib import Path; \
  print(Config(Path('config.json')).repositories)"

# Verify path exists
ls -la /path/to/repository
```

#### Ctags Not Working

**Symptom:** No symbols found during indexing

**Solution:**
```bash
# Verify ctags installation
ctags --version | grep "Universal Ctags"

# If "Exuberant Ctags" appears, wrong version
brew uninstall ctags
brew install universal-ctags
```

#### OAuth Authentication Failing

**Symptom:** `Invalid client credentials`

**Solution:**
```bash
# Regenerate OAuth credentials
python -m sigil_mcp.manage_auth regenerate-oauth

# Update client with new credentials
```

#### Slow Search Performance

**Symptom:** Search takes >5 seconds

**Solution:**
```bash
# Rebuild trigram index
python -c "
from sigil_mcp.indexer import SigilIndex
from pathlib import Path
index = SigilIndex(Path('~/.sigil_index').expanduser())
# Force rebuild
for repo in index.get_stats()['repos']:
    index.index_repository(repo['name'], Path(repo['path']), force=True)
"
```

#### High Memory Usage

**Symptom:** Python process using >2GB RAM

**Solution:**
- Reduce repository size or add skip patterns
- Disable vector embeddings if not needed
- Split large repositories into smaller logical units

### Log Analysis

#### Enable Debug Logging

```json
{
  "server": {
    "log_level": "DEBUG"
  }
}
```

#### Common Log Patterns

**Successful indexing:**
```
INFO: Indexed 342 files in my_project
INFO: Found 1,847 symbols
```

**File watching active:**
```
INFO: File watcher initialized
INFO: Watching repository: my_project
```

**Authentication success:**
```
INFO: OAuth token validated for client <id>
```

**Authentication failure:**
```
WARNING: Invalid OAuth token
```

### Getting Help

1. Check logs: `tail -f sigil.log`
2. Review documentation: `docs/`
3. Check GitHub issues: https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP/issues
4. Enable debug logging

---

## Monitoring & Health Checks

### Health Check Endpoint

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### MCP Ping Tool

From ChatGPT:
```
"Ping the Sigil server"
```

Response includes:
- Server status
- Configured repositories
- Index statistics

### Metrics to Monitor

1. **Server uptime** - Process running time
2. **Index size** - Disk usage of `~/.sigil_index`
3. **Search latency** - Time to complete searches
4. **Re-index frequency** - How often files are updated
5. **Authentication failures** - Potential security issues

### Basic Monitoring Script

```bash
#!/bin/bash
# sigil-monitor.sh

while true; do
    if curl -sf http://localhost:8000/health > /dev/null; then
        echo "[$(date)] Server healthy"
    else
        echo "[$(date)] Server DOWN - restarting"
        systemctl restart sigil-mcp
    fi
    sleep 60
done
```

---

## Backup & Recovery

### What to Back Up

1. **Configuration** - `config.json`, environment variables
2. **OAuth credentials** - From manage_auth
3. **Index data** - `~/.sigil_index/` (optional, can rebuild)
4. **API keys** - From manage_auth

### Backup Procedure

```bash
#!/bin/bash
# backup-sigil.sh

BACKUP_DIR=~/sigil-backups/$(date +%Y%m%d-%H%M%S)
mkdir -p $BACKUP_DIR

# Backup configuration
cp config.json $BACKUP_DIR/

# Backup OAuth data
python -m sigil_mcp.manage_auth show-oauth > $BACKUP_DIR/oauth-creds.txt

# Backup API keys
python -m sigil_mcp.manage_auth list-keys > $BACKUP_DIR/api-keys.txt

# Backup index (optional - large)
tar -czf $BACKUP_DIR/index.tar.gz ~/.sigil_index/

echo "Backup complete: $BACKUP_DIR"
```

### Recovery Procedure

```bash
# 1. Restore configuration
cp backup/config.json ./

# 2. Restore OAuth (if needed)
# Manual process - regenerate and update clients

# 3. Restore index (or rebuild)
tar -xzf backup/index.tar.gz -C ~/

# 4. Start server
python -m sigil_mcp.server
```

### Disaster Recovery

**Scenario:** Complete server loss

**Recovery steps:**
1. Install fresh Sigil MCP server
2. Restore `config.json`
3. Regenerate OAuth credentials
4. Re-index all repositories (index can't be restored)
5. Recreate API keys
6. Update all clients with new credentials

**Recovery time:** 30-60 minutes depending on repository size

---

## Performance Tuning

### Indexing Performance

**Optimize skip patterns:**
```json
{
  "index": {
    "skip_dirs": [
      "node_modules",
      "venv",
      ".git",
      "build",
      "dist",
      "coverage",
      "htmlcov"
    ]
  }
}
```

**Disable features you don't need:**
```json
{
  "embeddings": {
    "enabled": false
  },
  "watch": {
    "enabled": false
  }
}
```

**Enable embeddings with specific provider:**
```json
{
  "embeddings": {
    "enabled": true,
    "provider": "sentence-transformers",
    "model": "all-MiniLM-L6-v2",
    "dimension": 384
  }
}
```

See [EMBEDDING_SETUP.md](EMBEDDING_SETUP.md) for hardware-specific provider recommendations.

### Search Performance

**Use specific repository searches:**
```
"Search for 'async' in my_project"  # Good
"Search for 'async'"                 # Slower - searches all repos
```

**Use symbol search for definitions:**
```
"Find definition of HttpClient"     # Fast - uses symbol index
"Search for 'class HttpClient'"     # Slower - full text search
```

### File Watching Performance

**Increase debounce for bulk operations:**
```json
{
  "watch": {
    "debounce_seconds": 5.0
  }
}
```

**Add project-specific ignores:**
```json
{
  "watch": {
    "ignore_dirs": ["tmp", "cache", "logs"]
  }
}
```

### Resource Limits

**Expected resource usage:**
- **RAM:** 200-500MB base + 50MB per 1000 files indexed
- **Disk:** 100-200MB per 1000 files indexed
- **CPU:** Minimal idle, spikes during indexing/search

**Reduce memory usage:**
- Index fewer files
- Disable vector embeddings
- Restart server periodically

---

## Direct Python Testing

For testing the indexer directly without HTTP/MCP overhead, you can create a simple Python test script. This is useful for:
- Debugging indexing issues
- Performance testing
- CI/CD validation
- Development workflows

### Creating a Test Client

Create a file `test_client.py` in your project root:

```python
#!/usr/bin/env python3
"""
Direct test client for Sigil MCP Server indexer.
Tests indexing, search, and embeddings without HTTP layer.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sigil_mcp.indexer import SigilIndex
from sigil_mcp.config import get_config


def main():
    """Test main functionality."""
    config = get_config()
    
    print("=" * 60)
    print("SIGIL MCP SERVER TEST CLIENT")
    print("=" * 60)
    print()
    
    # Initialize index (without embeddings for basic testing)
    print("📊 Initializing index...")
    index = SigilIndex(
        config.index_path,
        embed_fn=None,
        embed_model="none"
    )
    print(f"   Index path: {config.index_path}")
    print()
    
    # List configured repositories
    print("📁 Configured Repositories:")
    for name, path in config.repositories.items():
        print(f"   - {name}: {path}")
    print()
    
    # Test indexing
    repo_name = list(config.repositories.keys())[0]  # First repo
    repo_path = Path(config.repositories[repo_name])
    
    print(f"🔍 Indexing repository: {repo_name}")
    print(f"   Path: {repo_path}")
    
    try:
        stats = index.index_repository(repo_name, repo_path, force=False)
        print(f"   ✅ Indexed: {stats.get('files_indexed', 0)} files")
        print(f"   📝 Symbols: {stats.get('symbols_extracted', 0)}")
        print(f"   ⏱️  Time: {stats.get('duration_seconds', 0):.2f}s")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 1
    print()
    
    # Test code search
    print("🔎 Testing code search for 'config'...")
    try:
        results = index.search_code(repo_name, "config", max_results=5)
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results[:3], 1):
            print(f"   {i}. {result.path}:{result.line}")
            snippet = result.text[:70].replace('\n', ' ')
            print(f"      {snippet}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()

    # Test symbol search
    print("🎯 Testing symbol search for 'Config'...")
    try:
        symbols = index.list_symbols(repo_name, "Config")
        print(f"   Found {len(symbols)} symbols:")
        for i, sym in enumerate(symbols[:5], 1):
            print(f"   {i}. {sym.name} ({sym.kind}) in {sym.file_path}:{sym.line}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Testing with Embeddings

To test vector embeddings and semantic search:

```python
#!/usr/bin/env python3
"""Test embeddings functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sigil_mcp.indexer import SigilIndex
from sigil_mcp.config import get_config
from sigil_mcp.embeddings import create_embedding_provider
import numpy as np


def main():
    config = get_config()
    
    if not config.embeddings_enabled:
        print("❌ Embeddings not enabled in config.json")
        return 1
    
    print(f"🧠 Testing Embeddings")
    print(f"   Provider: {config.embeddings_provider}")
    print(f"   Model: {config.embeddings_model}")
    print()
    
    # Create embedding provider
    try:
        provider = create_embedding_provider(
            provider=config.embeddings_provider,
            model=config.embeddings_model,
            dimension=config.embeddings_dimension,
            cache_dir=config.embeddings_cache_dir
        )
        
        def embed_fn(texts):
            embeddings = provider.embed_documents(texts)
            return np.array(embeddings, dtype="float32")
        
    except Exception as e:
        print(f"❌ Failed to initialize provider: {e}")
        return 1
    
    # Create index with embeddings
    index = SigilIndex(
        config.index_path,
        embed_fn=embed_fn,
        embed_model=str(config.embeddings_model)
    )
    
    # Build vector index
    repo_name = list(config.repositories.keys())[0]
    print(f"📊 Building vector index for {repo_name}...")
    
    try:
        vector_stats = index.build_vector_index(repo_name)
        print(f"   ✅ Chunks indexed: {vector_stats.get('chunks_indexed', 0)}")
        print(f"   📄 Documents: {vector_stats.get('documents_processed', 0)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 1
    print()
    
    # Test semantic search
    query = "configuration and settings"
    print(f"🔍 Semantic search: '{query}'")
    
    try:
        results = index.semantic_search(
            query=query,
            repo=repo_name,
            k=3
        )
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            path = result.get('path', 'unknown')
            start = result.get('start_line', 0)
            end = result.get('end_line', 0)
            score = result.get('score', 0.0)
            print(f"   {i}. {path}:{start}-{end}")
            print(f"      Score: {score:.3f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 1
    
    print()
    print("✅ Embeddings test complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Running Tests

```bash
# Basic test (no embeddings)
python test_client.py

# Test with embeddings
python test_embeddings.py

# Run with specific config
SIGIL_CONFIG=/path/to/config.json python test_client.py
```

### Expected Output

**Successful run:**
```
============================================================
SIGIL MCP SERVER TEST CLIENT
============================================================

📊 Initializing index...
   Index path: /home/user/.sigil_index

📁 Configured Repositories:
   - my_project: /path/to/my_project

🔍 Indexing repository: my_project
   Path: /path/to/my_project
   ✅ Indexed: 342 files
   📝 Symbols: 1847
   ⏱️  Time: 2.34s

🔎 Testing code search for 'config'...
   Found 12 results:
   1. config.py:45
      def load_config(path: Path) -> Dict[str, Any]:...
   2. server.py:12
      from sigil_mcp.config import get_config...

🎯 Testing symbol search for 'Config'...
   Found 5 symbols:
   1. Config (class) in config.py:34
   2. ConfigError (class) in config.py:89

============================================================
TEST COMPLETE
============================================================
```

### Troubleshooting Test Failures

**Import errors:**
```bash
# Ensure package is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/sigil-mcp-server:$PYTHONPATH
```

**No symbols found:**
```bash
# Check ctags installation
ctags --version | grep "Universal Ctags"

# If wrong version, reinstall
brew uninstall ctags && brew install universal-ctags
```

**Embedding errors:**
```bash
# Install embedding dependencies
pip install sentence-transformers

# Or use different provider
pip install openai  # For OpenAI embeddings
```

### Using Tests in CI/CD

**GitHub Actions example:**
```yaml
name: Test Indexer

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          sudo apt install universal-ctags
      
      - name: Create test config
        run: |
          cat > config.json <<EOF
          {
            "repositories": {
              "test_repo": "${{ github.workspace }}"
            }
          }
          EOF
      
      - name: Run tests
        run: python test_client.py
```

### Performance Benchmarking

Add timing to your test client:

```python
import time

# Time indexing
start = time.time()
stats = index.index_repository(repo_name, repo_path)
duration = time.time() - start

files_per_sec = stats.get('files_indexed', 0) / duration
print(f"   Performance: {files_per_sec:.1f} files/sec")
```

---

## Common Tasks

### Adding a New Repository

```bash
# 1. Edit config.json
{
  "repositories": {
    "new_repo": "/path/to/new/repo"
  }
}

# 2. Restart server
kill -TERM $(cat sigil.pid)
python -m sigil_mcp.server &

# 3. Index new repository
# From ChatGPT: "Index the new_repo repository"
```

### Removing a Repository

```bash
# 1. Remove from config.json
# (delete the repository entry)

# 2. Restart server

# 3. Clean up index (optional)
sqlite3 ~/.sigil_index/repos.db \
  "DELETE FROM repos WHERE name='old_repo'"
```

### Updating Repository Path

```bash
# 1. Update config.json with new path

# 2. Restart server

# 3. Force re-index
# From ChatGPT: "Force re-index repo_name"
```

### Upgrading Sigil MCP Server

```bash
# 1. Backup configuration and credentials
./backup-sigil.sh

# 2. Pull latest code
git pull origin main

# 3. Update dependencies
pip install -r requirements.txt --upgrade

# 4. Restart server
systemctl restart sigil-mcp

# 5. Verify functionality
curl http://localhost:8000/health

# 6. If upgrading from v0.3.0, rebuild index (path handling fixes)
rm -rf ~/.sigil_index
# Re-index via ChatGPT: "Index all repositories"
```

**Important: v0.3.0 → v0.3.1 Upgrade**

Version 0.3.1 includes critical path handling fixes. After upgrading from v0.3.0:

1. **Rebuild index completely** - Path handling changes require fresh index
2. **Test all tools** - Verify `list_repo_files`, `read_repo_file`, `search_repo` all work
3. **Check logs** - Ensure no "unsupported operand" or "no attribute 'rglob'" errors

### Debugging Search Issues

```bash
# Enable debug logging
export SIGIL_MCP_LOG_LEVEL=DEBUG

# Run search with verbose output
python -m sigil_mcp.server

# Check what's in the index
sqlite3 ~/.sigil_index/repos.db \
  "SELECT COUNT(*) FROM documents"
```

---

## Appendix

### File Locations

| Item | Default Location | Env Variable |
|------|-----------------|--------------|
| Config | `./config.json` | - |
| Index | `~/.sigil_index/` | `SIGIL_INDEX_PATH` |
| OAuth data | `~/.sigil_mcp/oauth.json` | - |
| API keys | `~/.sigil_mcp/api_keys.json` | - |
| Logs | stdout/stderr | - |

### Port Requirements

| Port | Purpose | Configurable |
|------|---------|--------------|
| 8000 | HTTP server | Yes (`SIGIL_MCP_PORT`) |

### External Dependencies

| Dependency | Required | Purpose |
|------------|----------|---------|
| Universal Ctags | Recommended | Symbol extraction |
| watchdog | Optional | File watching |
| sentence-transformers | Optional | Vector embeddings |

---

**Document Version:** 1.0  
**Maintained By:** Sigil MCP Development Team  
**Last Review:** 2025-12-03
