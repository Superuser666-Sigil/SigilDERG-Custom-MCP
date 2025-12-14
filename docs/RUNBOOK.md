<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Sigil MCP Server Operations Runbook

**Version:** 1.4  
**Last Updated:** 2025-12-10  
**Recommended Server Version:** v0.8.0 or later

This runbook provides operational procedures for running, troubleshooting, and maintaining the Sigil MCP Server in production and development environments.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation Procedures](#installation-procedures)
3. [Configuration Management](#configuration-management)
4. [Server Operations](#server-operations)
5. [Admin API](#admin-api)
6. [Index Management](#index-management)
7. [Authentication & Security](#authentication--security)
8. [File Watching](#file-watching)
9. [Direct Python Testing](#direct-python-testing)
10. [Troubleshooting](#troubleshooting)
11. [Monitoring & Health Checks](#monitoring--health-checks)
12. [Backup & Recovery](#backup--recovery)
13. [Performance Tuning](#performance-tuning)
14. [Common Tasks](#common-tasks)
15. [External MCP Servers](#external-mcp-servers)

---

## Quick Start

### Minimal Setup (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP.git
cd SigilDERG-Custom-MCP

# 2. Install dependencies (default stack: watcher + LanceDB + llama.cpp embeddings)
pip install -e .[server-full]

# 3. Configure repositories
cp config.example.json config.json
# Edit config.json with your repository paths

# 4. Start servers (MCP server + Admin UI)
./scripts/restart_servers.sh

# 5. Note OAuth credentials from startup output
#    Access Admin UI at http://localhost:5173
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

# For rocksdict trigram store (only backend; pulls rocksdict wheels; system RocksDB libs recommended)
pip install sigil-mcp-server[trigrams-rocksdb]

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
# Start all servers (MCP + Admin UI)
./scripts/restart_servers.sh

# Expected output:
# [SUCCESS] MCP Server started (PID: xxxxx)
# [SUCCESS] Frontend Dev Server started (PID: xxxxx)
# MCP Server:     http://127.0.0.1:8000
# Admin UI:       http://localhost:5173
#
# OAuth credentials will be in server logs:
# tail -f /tmp/sigil_server.log
```

**[WARNING] Important:** Save OAuth credentials securely!

---

## Configuration Management

### Configuration File Location

Default search order:
1. `./config.json` (current directory)
2. `~/.config/sigil-mcp/config.json`
3. Environment variables

### Run Mode (dev vs prod)

Set `mode` in `config.json` or `SIGIL_MCP_MODE` to control security defaults:
- `dev` (default): authentication off, local bypass allowed, admin API key optional.
- `prod`: authentication on, local bypass disabled, admin API requires `admin.api_key` and `admin.allowed_ips`.

The server logs warnings at startup if insecure overrides are present while `mode=prod`.

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

**Recommended: Use the restart script**

The `scripts/restart_servers.sh` script is the **primary entrypoint** for starting all server processes. It handles:
- Stopping any existing server processes
- Starting the MCP Server (port 8000)
- Starting the Admin UI frontend (port 5173)
- Running processes with `nohup` for persistence
- Port availability checks
- Process verification

```bash
# Start all servers (MCP + Admin UI)
./scripts/restart_servers.sh

# Stop all servers
./scripts/restart_servers.sh --stop
```

**What the script does:**
1. Finds and stops any running MCP Server or Frontend processes
2. Waits for ports 8000 and 5173 to be free
3. Starts MCP Server with `nohup` (logs to `/tmp/sigil_server.log`)
4. Starts Frontend Dev Server with `nohup` (logs to `/tmp/frontend.log`)
5. Verifies both servers started successfully
6. Displays server URLs and log file locations

**Manual start (MCP server only, for development):**

```bash
# Foreground (Development)
python -m sigil_mcp.server

# Background (Production)
nohup python -m sigil_mcp.server > sigil.log 2>&1 &
echo $! > sigil.pid

# Using systemd (recommended for production)
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

**Recommended: Use the restart script**

```bash
# Restart all servers (stops existing, then starts fresh)
./scripts/restart_servers.sh
```

**Manual restart:**

```bash
# Manual (MCP server only)
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

> [!IMPORTANT]
> **Using Cloudflare Tunnel Instead of ngrok?**  
> You must disable Cloudflare Bot Fight Mode or ChatGPT's OAuth authentication will fail.  
> 📖 See [**Cloudflare OAuth Issue & Solution**](CLOUDFLARE_OAUTH_ISSUE.md) for complete details.

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

## Admin API

The Admin API provides HTTP endpoints for operational management of the Sigil MCP Server. **It's integrated into the main MCP server process** (port 8000) and accessible at `/admin/*` endpoints. A React-based Admin UI is available on port 5173 for day-to-day use. Always start/stop the UI together with the MCP server so `/admin/*` calls terminate on the same host – the UI simply proxies API requests, so if you expose the server publicly ensure both ports are protected by the same OAuth/API-key rules.

### Starting the Admin API & UI

**Recommended: Use the restart script (starts everything):**

```bash
./scripts/restart_servers.sh
```

This starts:
- MCP Server on `http://127.0.0.1:8000` (includes Admin API at `/admin/*`)
- Admin UI on `http://localhost:5173`

**What the script does:**
- Stops any existing server processes
- Starts MCP Server with `nohup` (logs to `/tmp/sigil_server.log`)
- Starts Admin UI frontend with `nohup` (logs to `/tmp/frontend.log`)
- Verifies both servers started successfully
- Both processes persist after terminal closes

**Stop all servers:**
```bash
./scripts/restart_servers.sh --stop
```

### Configuration

The Admin API is enabled by default and integrated into the main server. Configure it in `config.json`:

```json
{
  "admin": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 8000,
    "api_key": null,
    "allowed_ips": ["127.0.0.1", "::1"]
  }
}
```

**Note:** The `admin.port` setting is legacy and no longer used - the Admin API runs on the same port as the MCP server (8000 by default).

Or via environment variables:
- `SIGIL_MCP_ADMIN_ENABLED` (default: "true")
- `SIGIL_MCP_ADMIN_REQUIRE_API_KEY` (default: "true" in prod mode, "true"/"false" respected in dev)
- `SIGIL_MCP_ADMIN_API_KEY` (optional, for additional security)
- `SIGIL_MCP_ADMIN_ALLOWED_IPS` (comma-separated list, default: "127.0.0.1,::1")

### Security

The Admin API uses two layers of security:

1. **IP Whitelist**: Only allows connections from configured IPs (default: localhost only)
2. **Optional API Key**: If `admin.api_key` is set, requires `X-Admin-Key` header

In **production mode**, the Admin API refuses requests unless `admin.api_key` is configured and the client IP is in `admin.allowed_ips`. CORS is limited to the shipped Admin UI origins; wildcard origins are never permitted.

**Example with API key:**
```bash
curl -H "X-Admin-Key: your-secret-key" http://127.0.0.1:8000/admin/status
```

**Example without API key (localhost only):**
```bash
curl http://127.0.0.1:8000/admin/status
```

### MCP Admin Endpoints
- `GET /admin/mcp/status` — external MCP aggregation status (servers, tools, last error)
- `POST /admin/mcp/refresh` — re-discover external MCP servers and re-register tools

### Endpoints

#### GET /admin/status

Get server status, repository list, index information, and watcher status.

```bash
curl http://127.0.0.1:8000/admin/status
```

Response:
```json
{
  "admin": {
    "host": "127.0.0.1",
    "port": 8000,
    "enabled": true
  },
  "repos": {
    "my-repo": "/path/to/repo"
  },
  "index": {
    "path": "/home/user/.sigil_index",
    "has_embeddings": true,
    "embed_model": "sentence-transformers:all-MiniLM-L6-v2"
  },
  "watcher": {
    "enabled": true,
    "watching": ["my-repo"]
  }
}
```

#### POST /admin/index/rebuild

Rebuild the trigram/symbol index for one or all repositories.

```bash
# Rebuild all repositories
curl -X POST http://127.0.0.1:8000/admin/index/rebuild \
  -H "Content-Type: application/json" \
  -d '{"force": true}'

# Rebuild specific repository
curl -X POST http://127.0.0.1:8000/admin/index/rebuild \
  -H "Content-Type: application/json" \
  -d '{"repo": "my-repo", "force": true}'
```

#### GET /admin/index/stats

Get index statistics for one or all repositories.

```bash
# All repositories
curl http://127.0.0.1:8000/admin/index/stats

# Specific repository
curl http://127.0.0.1:8000/admin/index/stats?repo=my-repo
```

#### POST /admin/vector/rebuild

Rebuild the vector embeddings index.

```bash
# Rebuild all repositories
curl -X POST http://127.0.0.1:8000/admin/vector/rebuild \
  -H "Content-Type: application/json" \
  -d '{"force": true}'

# Rebuild specific repository with custom model
curl -X POST http://127.0.0.1:8000/admin/vector/rebuild \
  -H "Content-Type: application/json" \
  -d '{"repo": "my-repo", "force": true, "model": "custom-model"}'
```

#### GET /admin/logs/tail

Get the last N lines from the server log file.

```bash
# Last 200 lines (default)
curl http://127.0.0.1:8000/admin/logs/tail

# Last 50 lines
curl http://127.0.0.1:8000/admin/logs/tail?n=50

# Last 1000 lines (max: 2000)
curl http://127.0.0.1:8000/admin/logs/tail?n=1000
```

#### GET /admin/config

View current configuration (read-only).

```bash
curl http://127.0.0.1:8000/admin/config
```

Returns the full `config.json` structure.

### Error Responses

All endpoints return structured error responses:

```json
{
  "error": "error_code",
  "detail": "Human-readable error message"
}
```

Common status codes:
- `200` - Success
- `401` - Unauthorized (invalid API key)
- `403` - Forbidden (IP not allowed)
- `404` - Not Found (log file not found)
- `500` - Internal Server Error
- `503` - Service Unavailable (admin API disabled)

### Use Cases

**Trigger index rebuild without restart:**
```bash
curl -X POST http://127.0.0.1:8000/admin/index/rebuild -d '{"force": true}'
```

**Check server health:**
```bash
curl http://127.0.0.1:8000/admin/status
```

**View recent errors:**
```bash
curl http://127.0.0.1:8000/admin/logs/tail?n=100 | grep ERROR
```

**Monitor index growth:**
```bash
watch -n 5 'curl -s http://127.0.0.1:8000/admin/index/stats | jq'
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
- `repos.db-wal` - Write-Ahead Log for concurrent access (v0.3.3+)
- `repos.db-shm` - Shared memory file for WAL mode (v0.3.3+)
- `trigrams.rocksdb/` - rocksdict trigram index (only supported backend)
- `blobs/` - Compressed file contents
- `lancedb/` - LanceDB vector store (one `code_vectors` table per repo)
- `vectors/` - Legacy vector embeddings (pre-LanceDB, if still present)

**Note:** RocksDB is used for trigram postings to reduce write contention. The `-wal` and `-shm` files are automatically managed by SQLite for the repos.db.

### Embeddings & LanceDB

- Install LanceDB support with `pip install -e .[lancedb]` or the bundled `.[server-full]` extra. If LanceDB is missing or `embeddings.enabled` is false, the server automatically falls back to trigram-only search and logs the reason.
- On startup and after rebuilds, the server logs whether the vector index is ready and how many chunks are indexed.
- `embeddings.provider` defaults to `llamacpp` with the Jina v2 code model; if a provider/model fails to load, embeddings are disabled and trigram search remains available.

### Thread Safety

Since v0.3.3, the indexer is fully thread-safe:

- **WAL Mode**: Multiple readers can query concurrently without blocking
- **RLock Serialization**: Write operations are serialized to prevent conflicts
- **File Watcher Safe**: Background re-indexing doesn't interfere with searches
- **Concurrent Searches**: Multiple MCP tool calls can run simultaneously

No special configuration needed - thread safety is automatic.

### Index Cleanup

```bash
# Remove index directory (requires rebuild). This is destructive.
rm -rf ~/.sigil_index

# Or specify a custom index path
rm -rf /path/to/custom/index
```

### Full Index Rebuild (Fresh Start)

For a **100% fresh setup** (all repositories, all blobs, trigrams, symbols, and embeddings):

```bash
# From the project root, use the helper script
python scripts/rebuild_indexes.py
```

This script:

- Deletes the entire index directory (default `~/.sigil_index`, or `index.path` from `config.json`)
- Recreates the index (including `lancedb/` for embeddings)
- Uses the default llama.cpp Jina model at `/home/dave/models/jina/jina-embeddings-v2-base-code-Q4_K_M.gguf` (768-dim) unless you override `embeddings.provider`/`embeddings.model`; ensure the file and llama.cpp runtime are available before running.
- Rebuilds all repositories defined in your current configuration

Use this when:

- You suspect index corruption and want a clean slate
- You have changed low-level indexing behavior and want all data regenerated
- You want to ensure no stale documents, trigrams, or vectors remain
- You're migrating from SQLite-backed embeddings and want LanceDB-only storage (drop the old table after the rebuild with `sqlite3 ~/.sigil_index/repos.db "DROP TABLE IF EXISTS embeddings;"`; only needed for legacy installs)

### Index Backup

```bash
# IMPORTANT: Checkpoint WAL before backup (v0.3.3+)
sqlite3 ~/.sigil_index/repos.db "PRAGMA wal_checkpoint(TRUNCATE)"

# Then backup entire index
tar -czf sigil-index-backup-$(date +%Y%m%d).tar.gz ~/.sigil_index

# Backup specific repository (also checkpoint first)
sqlite3 ~/.sigil_index/repos.db "PRAGMA wal_checkpoint(TRUNCATE)"
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
# [WARNING] This invalidates all existing tokens
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

**[WARNING] Never disable auth in production!**

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
#
# If you delete a file:
#   File deleted: test.py in my_project
#   Removed deleted file /path/to/project/test.py from index for repo my_project
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

### Header Logging

The server automatically logs all HTTP requests and responses via ASGI middleware. This provides visibility into:

- All incoming requests (MCP tool calls, OAuth, health checks)
- Request headers (with sensitive data redacted)
- Response status codes and duration
- Client IP addresses
- Cloudflare ray IDs (if using Cloudflare)

**Log Format:**

```
Incoming MCP HTTP request
  request_id=<uuid>
  method=POST
  path=/
  client_ip=203.0.113.42
  cf_ray=8978a4bf1c5a1234-DFW
  headers={...}

Outgoing MCP HTTP response
  request_id=<uuid>
  status_code=200
  duration_ms=143
  path=/
  method=POST
```

**Sensitive Headers Redacted:**
- `authorization`
- `cookie`
- `x-api-key`
- `x-admin-key`
- `x-openai-session`
- `x-openai-session-token`

**Using Logs for Debugging:**

```bash
# Find all requests from a specific client IP
grep "client_ip=203.0.113.42" server.log

# Find requests with errors
grep "status_code=5" server.log

# Find slow requests (>1 second)
grep "duration_ms=[0-9][0-9][0-9][0-9]" server.log

# Correlate request/response by request_id
grep "request_id=abc123" server.log

# Find requests with specific Cloudflare ray ID
grep "cf_ray=8978a4bf1c5a1234-DFW" server.log
```

**For Support Tickets:**

When reporting issues, include:
- Request ID (if available)
- Cloudflare ray ID (if using Cloudflare)
- Time window of the issue
- Relevant log entries (headers are already redacted)

## Monitoring & Health Checks

### Health & Readiness Endpoints

- Liveness: `curl http://localhost:8000/healthz` returns `{"status": "ok"}` without touching indexes.
- Readiness: `curl http://localhost:8000/readyz` returns 200 only after config loads, indexes open, and embeddings initialize (when enabled); otherwise 503 with component flags.

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
    print("[INFO] Initializing index...")
    index = SigilIndex(
        config.index_path,
        embed_fn=None,
        embed_model="none"
    )
    print(f"   Index path: {config.index_path}")
    print()
    
    # List configured repositories
    print("[INFO] Configured Repositories:")
    for name, path in config.repositories.items():
        print(f"   - {name}: {path}")
    print()
    
    # Test indexing
    repo_name = list(config.repositories.keys())[0]  # First repo
    repo_path = Path(config.repositories[repo_name])
    
    print(f"[INFO] Indexing repository: {repo_name}")
    print(f"   Path: {repo_path}")
    
    try:
        stats = index.index_repository(repo_name, repo_path, force=False)
        print(f"   [YES] Indexed: {stats.get('files_indexed', 0)} files")
        print(f"   [INFO] Symbols: {stats.get('symbols_extracted', 0)}")
        print(f"   [INFO] Time: {stats.get('duration_seconds', 0):.2f}s")
    except Exception as e:
        print(f"   [NO] Error: {e}")
        return 1
    print()
    
    # Test code search
    print("[INFO] Testing code search for 'config'...")
    try:
        results = index.search_code(repo_name, "config", max_results=5)
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results[:3], 1):
            print(f"   {i}. {result.path}:{result.line}")
            snippet = result.text[:70].replace('\n', ' ')
            print(f"      {snippet}...")
    except Exception as e:
        print(f"   [NO] Error: {e}")
    print()

    # Test symbol search
    print("[INFO] Testing symbol search for 'Config'...")
    try:
        symbols = index.list_symbols(repo_name, "Config")
        print(f"   Found {len(symbols)} symbols:")
        for i, sym in enumerate(symbols[:5], 1):
            print(f"   {i}. {sym.name} ({sym.kind}) in {sym.file_path}:{sym.line}")
    except Exception as e:
        print(f"   [NO] Error: {e}")
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
        print("[NO] Embeddings not enabled in config.json")
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
        print(f"[NO] Failed to initialize provider: {e}")
        return 1
    
    # Create index with embeddings
    index = SigilIndex(
        config.index_path,
        embed_fn=embed_fn,
        embed_model=str(config.embeddings_model)
    )
    
    # Build vector index
    repo_name = list(config.repositories.keys())[0]
    print(f" Building vector index for {repo_name}...")
    
    try:
        vector_stats = index.build_vector_index(repo_name)
        print(f"   [YES] Chunks indexed: {vector_stats.get('chunks_indexed', 0)}")
        print(f"    Documents: {vector_stats.get('documents_processed', 0)}")
    except Exception as e:
        print(f"   [NO] Error: {e}")
        return 1
    print()
    
    # Test semantic search
    query = "configuration and settings"
    print(f" Semantic search: '{query}'")
    
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
        print(f"   [NO] Error: {e}")
        return 1
    
    print()
    print("[YES] Embeddings test complete")
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

 Initializing index...
   Index path: /home/user/.sigil_index

[INFO] Configured Repositories:
   - my_project: /path/to/my_project

 Indexing repository: my_project
   Path: /path/to/my_project
   [YES] Indexed: 342 files
    Symbols: 1847
     Time: 2.34s

🔎 Testing code search for 'config'...
   Found 12 results:
   1. config.py:45
      def load_config(path: Path) -> Dict[str, Any]:...
   2. server.py:12
      from sigil_mcp.config import get_config...

 Testing symbol search for 'Config'...
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

## External MCP Servers

Sigil can register tools from other MCP servers and expose them with a server prefix (e.g., `playwright.click`, `mindsdb.query`).

1. Configure `external_mcp_servers` in `config.json` (see `config.example.json` and `docs/external_mcp.md`). Include auth headers for secured servers:
   ```json
   {
     "name": "playwright",
     "type": "streamable-http",
     "url": "http://127.0.0.1:3001/",
     "headers": { "authorization": "Bearer <token>" }
   }
   ```
2. Optional dev convenience: set `external_mcp_auto_install` to `true` (or `SIGIL_MCP_AUTO_INSTALL=true`) to run `npx`/`npm`/`bunx` or entries with `"auto_install": true` on startup. Disabled by default.
3. Tools are discovered at startup; names are prefixed by server. Diagnostics/tools:
   - `GET /admin/mcp/status` (admin access) for current servers/tools/errors
   - `POST /admin/mcp/refresh` to re-discover without restart
   - MCP tools: `list_mcp_tools`, `external_mcp_prompt`
4. Client preset: `docs/mcp.json` includes Sigil plus sample Playwright/Next.js MCP/MindsDB entries—update URLs/tokens to match your environment.

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

### Verifying the Admin UI Test Suite

When preparing a release or validating frontend changes, run the Admin UI regression suite to ensure coverage targets remain satisfied:

```bash
cd sigil-admin-ui
npm install
npm run test -- --coverage
```

Expectations:

1. Overall Vitest coverage ≥70%.
2. Critical flows (Status, Index, Vector, Logs, Config pages plus `src/utils/api.ts`) remain at 100% line coverage.
3. Failing tests usually indicate a missing `data-testid`, window/clipboard access outside guards, or timers not cleaned up. Resolve before shipping.

See [docs/adr-014-admin-ui-testing.md](adr-014-admin-ui-testing.md) for the testing strategy.

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
