<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Sigil MCP Server Troubleshooting Guide

**Version:** 1.1  
**Last Updated:** 2025-12-04

Quick reference for diagnosing and resolving common issues with Sigil MCP Server.

---

## Quick Diagnosis

```bash
# Check if server is running
curl http://localhost:8000/health

# Check logs
tail -50 sigil.log

# Check Admin API status (if enabled)
curl http://127.0.0.1:8000/admin/status

# View recent logs via Admin API
curl http://127.0.0.1:8000/admin/logs/tail?n=50

# Verify configuration
python -c "from sigil_mcp.config import Config; from pathlib import Path; \
  cfg = Config(Path('config.json')); print(cfg.repositories)"

# Check index
ls -lh ~/.sigil_index/
```

---

## Server Issues

### Server Won't Start

#### Symptom: Port Already in Use

```
Error: Address already in use: 127.0.0.1:8000
```

**Diagnosis:**
```bash
lsof -i :8000
# Shows process using port 8000
```

**Solution:**
```bash
# Kill existing process
kill -9 <PID>

# Or use different port
export SIGIL_MCP_PORT=8001
python -m sigil_mcp.server
```

#### Symptom: Module Not Found

```
ModuleNotFoundError: No module named 'sigil_mcp'
```

**Solution:**
```bash
# Install package
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Symptom: Configuration File Not Found

```
FileNotFoundError: config.json
```

**Solution:**
```bash
# Create config from example
cp config.example.json config.json

# Or specify path
export SIGIL_CONFIG_PATH=/path/to/config.json
```

### Server Crashes or Hangs

#### Symptom: Server Exits Immediately

**Diagnosis:**
```bash
# Run in foreground to see error
python -m sigil_mcp.server
```

**Common causes:**
- Invalid configuration JSON
- Permission denied on index directory
- Repository path doesn't exist

**Solution:**
```bash
# Validate JSON
python -m json.tool config.json

# Check permissions
ls -la ~/.sigil_index
mkdir -p ~/.sigil_index
chmod 755 ~/.sigil_index

# Verify repository paths
for repo in $(jq -r '.repositories[]' config.json); do
    test -d "$repo" && echo "[OK] $repo" || echo "[FAIL] $repo MISSING"
done
```

#### Symptom: High CPU Usage

**Diagnosis:**
```bash
top -p $(pgrep -f sigil_mcp)
```

**Common causes:**
- Large repository being indexed
- File watcher detecting too many changes
- Excessive search requests

**Solution:**
```bash
# Reduce file watching load
{
  "watch": {
    "debounce_seconds": 5.0,
    "ignore_dirs": ["node_modules", "build", "tmp"]
  }
}

# Add more skip patterns
{
  "index": {
    "skip_dirs": ["large_data", "archives"]
  }
}
```

#### Symptom: High Memory Usage

**Diagnosis:**
```bash
ps aux | grep sigil_mcp
# Check RSS (memory) column
```

**Common causes:**
- Large files in index
- Vector embeddings enabled
- Memory leak (rare)

**Solution:**
```bash
# Disable embeddings
{
  "embeddings": {
    "enabled": false
  }
}

# Restart periodically (cron job)
0 3 * * * systemctl restart sigil-mcp
```

---

## Indexing Issues

### Repository Not Indexing

#### Symptom: "0 files indexed"

**Diagnosis:**
```bash
# Check repository path
ls -la /path/to/repository

# Check skip patterns
cat config.json | jq '.index.skip_dirs'
```

**Common causes:**
- Incorrect repository path
- All files match skip patterns
- Permission denied

**Solution:**
```bash
# Verify path is correct
cd /path/to/repository && ls

# Temporarily disable skip patterns
{
  "index": {
    "skip_dirs": [],
    "skip_files": []
  }
}

# Check file permissions
find /path/to/repository -type f ! -readable
```

### Vector embeddings missing after upgrade

**Symptom:** `semantic_search` returns no results or logs show `code_vectors` not found.

**Diagnosis:**
```bash
ls -la ~/.sigil_index/lancedb
# (Legacy installs only) check for old SQLite embeddings table
sqlite3 ~/.sigil_index/repos.db "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings';"
```

**Solution:**
1. Rebuild embeddings into LanceDB: `python scripts/rebuild_indexes.py` (or `POST /admin/vector/rebuild` per repo).
2. If the legacy SQLite `embeddings` table exists, drop it to avoid confusion (legacy only): `sqlite3 ~/.sigil_index/repos.db "DROP TABLE IF EXISTS embeddings;"`.
3. Confirm per-repo subdirectories exist under `~/.sigil_index/lancedb/`.

### Slow Indexing

#### Symptom: Indexing takes >5 minutes for small repo

**Diagnosis:**
```bash
# Enable debug logging
export SIGIL_MCP_LOG_LEVEL=DEBUG
python -m sigil_mcp.server
```

**Common causes:**
- Large binary files
- Network-mounted filesystem
- Ctags processing large files

**Solution:**
```bash
# Skip binary files
{
  "index": {
    "skip_files": ["*.pdf", "*.zip", "*.tar.gz", "*.bin"]
  }
}

# Disable ctags temporarily
# (symbols won't be extracted)
export SIGIL_DISABLE_CTAGS=true
```

### Symbols Not Found

#### Symptom: "No symbols found" or search finds text but not symbols

**Diagnosis:**
```bash
# Check ctags installation
ctags --version | grep "Universal Ctags"

# Test ctags on a file
ctags -f - --fields=+n --output-format=json /path/to/file.py
```

**Common causes:**
- Ctags not installed
- Wrong ctags version (Exuberant instead of Universal)
- Language not supported

**Solution:**
```bash
# macOS
brew uninstall ctags
brew install universal-ctags

# Linux
sudo apt remove exuberant-ctags
sudo apt install universal-ctags

# Verify
ctags --version
```

### Index Corruption

#### Symptom: SQLite errors, missing data, or crashes

```
sqlite3.DatabaseError: database disk image is malformed
database is locked
```

**Common Causes:**
- Disk full or I/O errors during write
- Unclean shutdown (power loss, kill -9)
- Network filesystem issues (if index on NFS)
- ~~Concurrent access without thread safety~~ (fixed in v0.3.3+)

**Solution:**
```bash
# 1. Check for WAL mode (v0.3.3+)
sqlite3 ~/.sigil_index/repos.db "PRAGMA journal_mode"
# Should return: wal

# 2. If corruption is detected or you want a completely fresh index,
#    move the old directory out of the way
mv ~/.sigil_index ~/.sigil_index.corrupt

# 3. From the project root, run the rebuild script
cd /path/to/sigil-mcp-server
python scripts/rebuild_indexes.py

# This will:
#   - Delete any existing index at the configured path
#   - Recreate the index
#   - Rebuild all configured repositories from scratch
```

**Note:** Since v0.3.3, the indexer uses WAL mode and threading locks to prevent concurrent access issues. If you still see "database is locked" errors, check:
- Index is not on a network filesystem (NFS, SMB)
- No other processes accessing the database
- Sufficient disk space for WAL files

---

## Search Issues

### No Search Results

#### Symptom: Search returns empty results for known code

**Diagnosis:**
```bash
# Check if repository is indexed
sqlite3 ~/.sigil_index/repos.db \
  "SELECT name, indexed_at FROM repos"

# Check if specific file is indexed
sqlite3 ~/.sigil_index/repos.db \
  "SELECT path FROM documents WHERE path LIKE '%filename%'"
```

**Common causes:**
- Repository not indexed
- File skipped by skip patterns
- Search term too short (trigrams need 3+ chars)

**Solution:**
```bash
# Re-index repository
# From ChatGPT: "Force re-index repo_name"

# Check skip patterns exclude the file
cat config.json | jq '.index.skip_files'

# For short terms, use symbol search instead
"Find definition of DB"  # Uses symbol index
```

### Slow Search

#### Symptom: Search takes >5 seconds

**Diagnosis:**
```bash
# Check index size (approximate)
ls -la ~/.sigil_index/trigrams.rocksdb/

# Check repository count
wc -l config.json | jq '.repositories | length'
```

**Common causes:**
- Too many repositories indexed
- Searching across all repos instead of one
- Trigram index not optimized

**Solution:**
```bash
# Search specific repository
"Search for 'async' in project_name"

# Rebuild trigram index (rocksdict is the only supported backend)
rm -rf ~/.sigil_index/trigrams.rocksdb
# From ChatGPT: "Re-index all repositories"

# Reduce indexed repositories
# Remove unused repos from config.json
```

### Inaccurate Results

#### Symptom: Results don't match expected files

**Common causes:**
- Stale index (file changes while file watching disabled)
- Wrong repository selected
- Case sensitivity

**Solution:**
```bash
# Enable file watching to keep index current; deletions and modifications will then
# be applied incrementally via granular indexing + remove_file.
pip install sigil-mcp-server[watch]
{
  "watch": {
    "enabled": true
  }
}

# If file watching was disabled for a while and you suspect drift,
# perform a force re-index for the affected repo:
"Force re-index repo_name"

# For extreme cases (e.g., index corruption or major indexing changes),
# run the full rebuild script from the project root:
python scripts/rebuild_indexes.py

# Be specific with repository name when searching
"Search for 'function' in exact_repo_name"
```

---

## Authentication Issues

### OAuth Authentication Failing

> [!IMPORTANT]
> **ChatGPT OAuth Failing with Cloudflare Tunnel?**  
> This is a known issue caused by Cloudflare Bot Fight Mode blocking ChatGPT's backend.  
> 📖 See [**Cloudflare OAuth Issue & Solution**](CLOUDFLARE_OAUTH_ISSUE.md) for the fix.

> **502 Bad Gateway with Cloudflare Tunnel?**  
> Cloudflare may buffer responses, causing 502 errors. The server automatically adds headers to prevent this.  
> 📖 See [**Cloudflare 502 Fix Guide**](CLOUDFLARE_502_FIX.md) for details and troubleshooting.

#### Symptom: "Invalid client credentials"

**Diagnosis:**
```bash
# Check OAuth credentials
python -m sigil_mcp.manage_auth show-oauth
```

**Common causes:**
- Credentials not configured in client
- Credentials regenerated
- Clock skew (token expiry)

**Solution:**
```bash
# Show current credentials
python -m sigil_mcp.manage_auth show-oauth

# If lost, regenerate (invalidates all tokens)
python -m sigil_mcp.manage_auth regenerate-oauth

# Update client with new credentials
```

#### Symptom: "Token expired"

**Solution:**
```bash
# OAuth tokens expire after 1 hour
# Client should automatically refresh

# Check token expiry
# (requires looking at token claims)

# Force new token by re-authenticating
```

#### Symptom: "Forbidden" from localhost

**Diagnosis:**
```bash
# Check local bypass setting
cat config.json | jq '.authentication.allow_local_bypass'
```

**Solution:**
```bash
# Enable local bypass
{
  "authentication": {
    "allow_local_bypass": true
  }
}

# Restart server
```

### API Key Issues

#### Symptom: "Invalid API key"

**Diagnosis:**
```bash
# List valid keys
python -m sigil_mcp.manage_auth list-keys
```

**Common causes:**
- Key expired
- Key revoked
- Typo in key

**Solution:**
```bash
# Create new key
python -m sigil_mcp.manage_auth create-key --name "New Key" --expires 365

# Use full key in Authorization header
# Authorization: Bearer sk_...
```

---

## File Watching Issues

### File Watching Not Working

#### Symptom: File changes don't trigger re-indexing

**Diagnosis:**
```bash
# Check if watchdog is installed
python -c "import watchdog; print('[OK] watchdog installed')"

# Check config
cat config.json | jq '.watch.enabled'

# Check logs
tail -f sigil.log | grep -i watch
```

**Common causes:**
- Watchdog not installed
- File watching disabled in config
- File matches ignore pattern

**Solution:**
```bash
# Install watchdog
pip install watchdog>=3.0.0

# Enable watching
{
  "watch": {
    "enabled": true
  }
}

# Check ignore patterns
cat config.json | jq '.watch.ignore_extensions'

# Restart server
```

#### Symptom: "watchdog not available" warning

**Solution:**
```bash
# Install optional dependency
pip install 'sigil-mcp-server[watch]'

# Or install directly
pip install watchdog>=3.0.0
```

#### Symptom: Excessive re-indexing

**Diagnosis:**
```bash
# Watch log for re-index events
tail -f sigil.log | grep "Re-indexed"
```

**Common causes:**
- Low debounce time
- IDE or tool generating temp files
- Build process creating many files

**Solution:**
```bash
# Increase debounce
{
  "watch": {
    "debounce_seconds": 5.0
  }
}

# Add ignore patterns
{
  "watch": {
    "ignore_dirs": ["tmp", ".cache", "build"],
    "ignore_extensions": [".tmp", ".swp", ".swo"]
  }
}
```

---

## Vector Embeddings Issues

### Embeddings Not Working

#### Symptom: "Embeddings not available"

**Diagnosis:**
```bash
# Check if dependencies installed
python -c "import sentence_transformers; print('[OK]')"
```

**Solution:**
```bash
# Install embeddings dependencies
pip install 'sigil-mcp-server[embeddings]'

# Enable in config
{
  "embeddings": {
    "enabled": true
  }
}
```

### Slow Embedding Generation

#### Symptom: Initial indexing takes hours

**Common causes:**
- Large repository
- CPU-only (no GPU)
- Large chunk size

**Solution:**
```bash
# Use smaller model
{
  "embeddings": {
    "model": "all-MiniLM-L6-v2"  # Faster, smaller
  }
}

# Reduce chunk size
{
  "embeddings": {
    "chunk_size": 256  # Default is 512
  }
}

# Disable for large repos
{
  "embeddings": {
    "enabled": false
  }
}
```

---

## Network Issues

### Can't Connect from External Client

#### Symptom: Connection refused from remote client

**Diagnosis:**
```bash
# Check server is listening on correct interface
netstat -tlnp | grep 8000
# Should show 0.0.0.0:8000 for external access
# Or specific IP address
```

**Common causes:**
- Server listening on 127.0.0.1 only
- Firewall blocking port
- ngrok/proxy not configured

**Solution:**
```bash
# Listen on all interfaces
{
  "server": {
    "host": "0.0.0.0"
  }
}

# Or use ngrok
ngrok http 8000

# Check firewall
sudo ufw allow 8000
```

#### Symptom: HTTPS errors with ngrok

**Common causes:**
- ChatGPT requires HTTPS
- Invalid ngrok URL

**Solution:**
```bash
# Use ngrok's https URL
ngrok http 8000
# Use: https://abc123.ngrok.io (not http)

# Verify SSL
curl -v https://your-url.ngrok.io/health
```

#### Symptom: Path Handling Errors

```
TypeError: unsupported operand type(s) for /: 'str' and 'str'
AttributeError: 'str' object has no attribute 'rglob'
```

**Cause:**
Repository paths stored as strings in configuration aren't being converted to Path objects before file operations.

**Version Affected:** v0.3.0 and earlier

**Solution (Fixed in v0.3.1):**

Upgrade to v0.3.1 or later:
```bash
git pull origin main
pip install -e .
```

**Manual Fix (if needed):**

The issue occurs when repository paths from `config.json` or environment variables are strings but Path operations are expected. The fix ensures `_get_repo_root()` always returns a Path object:

```python
# sigil_mcp/server.py
def _get_repo_root(name: str) -> Path:
    try:
        root = REPOS[name]
        # Ensure we return a Path object
        if isinstance(root, str):
            return Path(root)
        return root
    except KeyError:
        raise ValueError(f"Unknown repo {name!r}")
```

**After upgrading, rebuild index:**
```bash
# Remove old index
rm -rf ~/.sigil_index

# Start server (will create fresh index)
python -m sigil_mcp.server

# Re-index all repositories
# From ChatGPT: "Index all repositories"
```

#### Symptom: "Invalid Host header" or "Invalid Content-Type" with ChatGPT

```
INFO: Created new transport with session ID: abc123...
Invalid Content-Type header: application/octet-stream
INFO: "POST / HTTP/1.1" 400 Bad Request
```

**Cause:**
ChatGPT's MCP connector is **not fully compliant** with the MCP streamable-http specification:
1. Sends `Content-Type: application/octet-stream` instead of `application/json`
2. Sends ngrok Host headers that trigger DNS rebinding protection

**Solution (Already Applied):**

The server **disables DNS rebinding protection** for ChatGPT compatibility:

```python
# In sigil_mcp/server.py
transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False
)

mcp = FastMCP(
    name=config.server_name,
    json_response=True,
    streamable_http_path="/",
    transport_security=transport_security
)
```

**Security Tradeoff:**
- [NO] DNS rebinding protection: Disabled (Host header not validated)
- [NO] Content-Type validation: Disabled (accepts application/octet-stream)
- [YES] OAuth 2.0 authentication: Still active and required
- [YES] Bearer token validation: Still active
- [YES] Token expiration: Still enforced

**Why This Was Necessary:**

ChatGPT's MCP implementation doesn't follow the standard:
- **MCP Spec requires**: `Content-Type: application/json`
- **ChatGPT sends**: `Content-Type: application/octet-stream`
- **MCP Spec requires**: Valid Host header matching server
- **ChatGPT sends**: ngrok domain triggering rebinding protection

Disabling these validations allows ChatGPT to connect while maintaining OAuth authentication.

**Verification:**
```bash
# Server should accept ChatGPT requests
tail -f /tmp/sigil_server.log | grep -E "POST|OAuth"

# Should see successful OAuth and POST requests
```

**References:**
- [MCP Streamable HTTP Spec](https://modelcontextprotocol.io/specification/2025-06-18/transport/streamable-http)
- [FastMCP TransportSecuritySettings](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/transport_security.py)

---

## Debugging with Header Logs

The server automatically logs all HTTP requests and responses with redacted headers. This is invaluable for debugging client issues.

### Viewing Request Logs

```bash
# View all incoming requests
grep "Incoming MCP HTTP request" server.log

# Find requests from specific IP
grep "client_ip=203.0.113.42" server.log

# Find requests with errors (5xx status)
grep "status_code=5" server.log

# Find slow requests (>1 second)
grep "duration_ms=[0-9][0-9][0-9][0-9]" server.log

# Correlate request/response by request_id
grep "request_id=abc123-def456" server.log

# Find requests with specific Cloudflare ray ID
grep "cf_ray=8978a4bf1c5a1234-DFW" server.log
```

### Using Admin API for Debugging

```bash
# Check server status
curl http://127.0.0.1:8000/admin/status

# View recent logs (last 100 lines)
curl http://127.0.0.1:8000/admin/logs/tail?n=100

# Check index statistics
curl http://127.0.0.1:8000/admin/index/stats

# View configuration
curl http://127.0.0.1:8000/admin/config
```

### Common Debugging Scenarios

**ChatGPT connection issues:**
```bash
# Find all ChatGPT requests
grep "user-agent.*OpenAI" server.log

# Find failed OAuth requests
grep "OAuth.*request" server.log | grep "status_code=4"

# Find requests with specific error
grep "Error handling MCP request" server.log
```

**Performance issues:**
```bash
# Find slow requests
grep "duration_ms" server.log | awk -F'duration_ms=' '{print $2}' | sort -n | tail -10

# Find requests to specific endpoint
grep "path=/oauth" server.log
```

**For support tickets, include:**
- Request ID (if available)
- Cloudflare ray ID (if using Cloudflare)
- Time window of the issue
- Relevant log entries (headers are already redacted)

---

## Performance Issues

### General Slowness

**Diagnosis checklist:**
```bash
# 1. Check system resources
top
df -h

# 2. Check index size
du -sh ~/.sigil_index

# 3. Check repository sizes
du -sh /path/to/repos/*

# 4. Check number of files indexed
sqlite3 ~/.sigil_index/repos.db \
  "SELECT COUNT(*) FROM documents"
```

**Optimization steps:**

1. **Add skip patterns** for large unnecessary files
2. **Disable embeddings** if not using semantic search
3. **Disable file watching** if not actively developing
4. **Split large repositories** into smaller logical units
5. **Increase debounce time** for file watching
6. **Run on SSD** instead of HDD

---

## Data Corruption or Loss

### Lost OAuth Credentials

**Impact:** Clients can't authenticate

**Recovery:**
```bash
# Generate new credentials
python -m sigil_mcp.manage_auth regenerate-oauth

# Update all clients with new credentials
# Note: This invalidates all existing tokens
```

### Lost Index Data

**Impact:** No search results, must rebuild

**Recovery:**
```bash
# Index will be automatically created if missing
python -m sigil_mcp.server

# Re-index all repositories
# From ChatGPT: "Index all repositories"
```

### Corrupted Configuration

**Impact:** Server won't start

**Recovery:**
```bash
# Backup corrupted config
mv config.json config.json.corrupt

# Restore from template
cp config.example.json config.json

# Edit with your repositories
nano config.json
```

---

## Getting Additional Help

### Enabling Debug Mode

```bash
# Set log level to DEBUG
export SIGIL_MCP_LOG_LEVEL=DEBUG

# Or in config.json
{
  "server": {
    "log_level": "DEBUG"
  }
}

# Restart server and reproduce issue
python -m sigil_mcp.server
```

### Collecting Diagnostic Information

```bash
#!/bin/bash
# collect-diagnostics.sh

echo "=== System Info ===" > diagnostics.txt
uname -a >> diagnostics.txt
python --version >> diagnostics.txt

echo -e "\n=== Sigil Version ===" >> diagnostics.txt
pip show sigil-mcp-server >> diagnostics.txt

echo -e "\n=== Dependencies ===" >> diagnostics.txt
pip list | grep -E "(mcp|numpy|watchdog|sentence)" >> diagnostics.txt

echo -e "\n=== Config ===" >> diagnostics.txt
cat config.json >> diagnostics.txt

echo -e "\n=== Index Stats ===" >> diagnostics.txt
du -sh ~/.sigil_index >> diagnostics.txt
sqlite3 ~/.sigil_index/repos.db "SELECT COUNT(*) FROM documents" >> diagnostics.txt

echo -e "\n=== Recent Logs ===" >> diagnostics.txt
tail -100 sigil.log >> diagnostics.txt

echo "Diagnostics saved to: diagnostics.txt"
```

### Reporting Issues

When reporting issues, include:

1. **Error message** - Full text including traceback
2. **Steps to reproduce** - What actions trigger the issue
3. **Configuration** - Sanitized config.json (remove sensitive paths)
4. **Environment** - OS, Python version, dependency versions
5. **Logs** - Last 50-100 lines with DEBUG level enabled
6. **Expected vs actual behavior**

GitHub Issues: https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP/issues

---

**Document Version:** 1.0  
**Maintained By:** Sigil MCP Development Team  
**Last Review:** 2025-12-03
