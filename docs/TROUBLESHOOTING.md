<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Sigil MCP Server Troubleshooting Guide

**Version:** 1.0  
**Last Updated:** 2025-12-03

Quick reference for diagnosing and resolving common issues with Sigil MCP Server.

---

## Quick Diagnosis

```bash
# Check if server is running
curl http://localhost:8000/health

# Check logs
tail -50 sigil.log

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
    test -d "$repo" && echo "✓ $repo" || echo "✗ $repo MISSING"
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
```

**Solution:**
```bash
# Backup corrupted index
mv ~/.sigil_index ~/.sigil_index.corrupt

# Rebuild from scratch
mkdir ~/.sigil_index
python -m sigil_mcp.server

# Re-index all repositories
# From ChatGPT: "Index all repositories"
```

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
# Check index size
sqlite3 ~/.sigil_index/trigrams.db \
  "SELECT COUNT(*) FROM trigrams"

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

# Rebuild trigram index
rm ~/.sigil_index/trigrams.db
# From ChatGPT: "Re-index all repositories"

# Reduce indexed repositories
# Remove unused repos from config.json
```

### Inaccurate Results

#### Symptom: Results don't match expected files

**Common causes:**
- Stale index (files changed)
- Wrong repository selected
- Case sensitivity

**Solution:**
```bash
# Enable file watching to keep index current
pip install sigil-mcp-server[watch]
{
  "watch": {
    "enabled": true
  }
}

# Force re-index
"Force re-index repo_name"

# Be specific with repository name
"Search for 'function' in exact_repo_name"
```

---

## Authentication Issues

### OAuth Authentication Failing

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
python -c "import watchdog; print('✓ watchdog installed')"

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
python -c "import sentence_transformers; print('✓')"
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
