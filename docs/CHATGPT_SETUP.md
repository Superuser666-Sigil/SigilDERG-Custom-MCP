<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ChatGPT Setup Guide for Sigil MCP Server

This guide shows you how to configure your Sigil MCP server with ChatGPT using Developer Mode and a custom connector.

Semantic search now stores embeddings in a LanceDB vector store at `~/.sigil_index/lancedb/` (or your configured index path).
If you're upgrading from a build that used the SQLite `embeddings` table, rebuild vectors once and drop the old table to keep
ChatGPT results consistent: `python scripts/rebuild_indexes.py` then (legacy installs only) `sqlite3 ~/.sigil_index/repos.db "DROP TABLE IF EXISTS embeddings;"`.

## Prerequisites

- ChatGPT Plus subscription
- Developer Mode enabled in ChatGPT
- ngrok installed and configured
- Sigil MCP server v0.3.1 or later (includes critical path handling fixes)
- Authentication enabled on server

## Overview

ChatGPT can connect to your Sigil MCP server in two ways:

1. **Via Developer Mode** - Configure MCP server URL directly in ChatGPT settings (recommended for development)
2. **Via Custom Connector** - Create a custom GPT with actions (for production/sharing)

Both methods support the **API Key authentication** we implemented.

---

## Method 1: Developer Mode Configuration (Recommended)

Developer Mode in ChatGPT Plus allows you to connect to remote MCP servers directly without creating custom actions.

### Step 1: Start Your Server and ngrok

```bash
# Terminal 1: Start Sigil server
cd /home/dave/dev/sigil-mcp-server
python -m sigil_mcp.server
```

**Important:** ChatGPT's MCP connector has compatibility issues with standard MCP security:

1. **Content-Type Issue**: ChatGPT sends `application/octet-stream` instead of `application/json`
2. **Host Header Issue**: ngrok domains trigger DNS rebinding protection

**Solution**: The server **disables DNS rebinding protection** for ChatGPT compatibility:

```python
# In server.py
transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False
)

mcp = FastMCP(
    name=config.server_name,
    json_response=True,
    streamable_http_path="/",  # Mount at root for ChatGPT
    transport_security=transport_security
)
```

**Security Note**: This disables Host header and Content-Type validation. Your other security layers remain active:
- [YES] OAuth 2.0 authentication with PKCE
- [YES] Bearer token validation
- [YES] Token expiration and refresh
- [NO] DNS rebinding protection (disabled)
- [NO] Content-Type validation (disabled)

```bash
# Terminal 2: Start ngrok tunnel
ngrok http 8000
```

Copy the ngrok HTTPS URL (e.g., `https://abc123.ngrok-free.app`).

### Step 2: Configure in ChatGPT

1. Go to ChatGPT Settings → Developer Mode → MCP Servers
2. Click "Add MCP Server"
3. Configure as follows:

**Server Configuration:**
```json
{
  "name": "Sigil Code Server",
  "server_url": "https://YOUR-NGROK-URL.ngrok-free.app",
  "authorization_url": "https://YOUR-NGROK-URL.ngrok-free.app/oauth/authorize",
  "token_url": "https://YOUR-NGROK-URL.ngrok-free.app/oauth/token",
  "client_id": "YOUR_CLIENT_ID",
  "client_secret": "YOUR_CLIENT_SECRET",
  "description": "Local code repository indexer with IDE-like features"
}
```

**Note**: The MCP endpoint is at the **root path** (`/`), not `/mcp`. This is required for ChatGPT compatibility.

Replace:
- `YOUR-NGROK-URL.ngrok-free.app` with your actual ngrok URL
- `YOUR_API_KEY_HERE` with the API key displayed when you started the server

### Step 3: Test the Connection

In ChatGPT, try:
- "Index my project at /path/to/my/repo"
- "Search for 'function handleRequest'"
- "Show me all classes in the codebase"
- "Find the definition of MyClass"

---

## Method 2: Custom Actions (Alternative Approach)

If Developer Mode doesn't support your authentication needs, you can create a custom GPT with actions.

### Step 1: Create OpenAPI Schema

ChatGPT Actions require an OpenAPI 3.0+ schema. Create this file to define your API:

**Save as `sigil-openapi.yaml`:**

```yaml
openapi: 3.0.0
info:
  title: Sigil MCP Server
  description: Local code repository indexer with IDE-like search and navigation
  version: 1.0.0
servers:
  - url: https://YOUR-NGROK-URL.ngrok-free.app
    description: Sigil MCP Server via ngrok

paths:
  /tools/call:
    post:
      summary: Call MCP tools
      description: Execute MCP tools like search_code, goto_definition, list_symbols, etc.
      operationId: callMCPTool
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                name:
                  type: string
                  description: Name of the tool to call
                  enum: [search_code, goto_definition, list_symbols, get_index_stats, index_repository]
                arguments:
                  type: object
                  description: Tool-specific arguments
              required:
                - name
                - arguments
      responses:
        '200':
          description: Tool execution result
          content:
            application/json:
              schema:
                type: object
                properties:
                  result:
                    type: object
      security:
        - ApiKeyAuth: []

components:
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
```

**Important:** Replace `YOUR-NGROK-URL.ngrok-free.app` with your actual ngrok URL.

### Step 2: Create Custom GPT with Actions

1. Go to [https://chatgpt.com/gpts/editor](https://chatgpt.com/gpts/editor)
2. Click "Create" tab and describe your GPT:
   ```
   You are a code navigation assistant that helps developers search, explore, and 
   understand their local codebases. You can index repositories, search for code 
   patterns, find symbol definitions, and list all symbols in a project.
   ```

3. Switch to "Configure" tab:
   - **Name:** Sigil Code Navigator
   - **Description:** Search and navigate local code repositories with IDE-like features
   - **Instructions:** 
     ```
     You help developers work with their local code repositories. You can:
     - Index repositories to make them searchable
     - Search for code using substring or regex patterns
     - Find definitions of functions, classes, and methods
     - List all symbols in a project
     - Provide code statistics
     
     Always ask for the repository path before indexing. Be helpful and precise 
     when showing search results or code locations.
     ```

4. **Enable Actions:**
   - Scroll to "Actions" section
   - Click "Create new action"
   - Click "Import from URL" or paste the OpenAPI schema
   - If using file: Upload your `sigil-openapi.yaml`

### Step 3: Configure Authentication

In the Actions configuration:

1. **Authentication Type:** Select "API Key"
2. **API Key Configuration:**
   - **Auth Type:** Custom
   - **Custom Header Name:** `X-API-Key`
   - **API Key:** Paste your Sigil API key (from server startup)

ChatGPT encrypts and stores your API key securely. You won't need to provide it again.

### Step 4: Test Your Custom GPT

Click "Test" in the GPT editor and try:
- "Index the repository at /home/dave/dev/myproject"
- "Search for all TODO comments"
- "Find the definition of UserController"

### Step 5: Publish (Optional)

- Click "Publish" 
- Choose "Only me" for private use
- Or "Anyone with link" to share with your team

---

## API Key Authentication Details

### How It Works

Your Sigil MCP server validates every request using the API key:

1. Client sends request with header: `X-API-Key: YOUR_API_KEY`
2. Server hashes the provided key using SHA-256
3. Server compares hashed key with stored hash (timing-attack resistant)
4. Request proceeds only if keys match

### Security Notes

- **HTTPS:** ngrok provides TLS encryption automatically (even free tier)
- **API Key:** Hashed using SHA-256, stored in `~/.sigil_mcp_server/api_key`
- **No plaintext:** The server never stores your API key in plaintext
- **ChatGPT storage:** OpenAI encrypts API keys when storing them

### Managing API Keys

```bash
# View current API key status
./manage_auth.py show

# Generate a new API key (invalidates old one)
./manage_auth.py generate

# Reset everything (delete and regenerate)
./manage_auth.py reset
```

After regenerating, you must update the key in ChatGPT Developer Mode or your Custom GPT Actions.

---

## Troubleshooting

### "Invalid Host header" error

**Problem:** Server returns HTTP 421 with "Invalid Host header: abc123.ngrok-free.app" in logs.

**Cause:** FastMCP's DNS rebinding protection validates the `Host` header. By default, it blocks all non-localhost hostnames including ngrok.

**Solution:**
Add `allowed_hosts` configuration in `config.json`:
```json
{
  "server": {
    "allowed_hosts": ["*"]
  }
}
```

Restart the server:
```bash
python -m sigil_mcp.server
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#symptom-invalid-host-header-with-ngrok-or-chatgpt) for more details.

### "Authentication failed" error

**Problem:** ChatGPT can't authenticate with your server.

**Solutions:**
1. Check OAuth credentials are correctly configured
2. Verify ngrok URL is correct (changes on restart)
3. Ensure server is running with authentication enabled
4. View server logs for authentication errors

### "Connection refused" or "Server unreachable"

**Problem:** ChatGPT can't reach your server.

**Solutions:**
1. Verify ngrok is running: `ngrok http 8000`
2. Check that server is running: `ps aux | grep sigil_mcp`
3. Test connection manually: `curl https://YOUR-NGROK-URL.ngrok-free.app/health`
4. Ensure ngrok URL in ChatGPT matches current tunnel URL (changes on restart)

### Tools not available in ChatGPT

**Problem:** ChatGPT doesn't see your MCP tools.

**Solutions:**
1. Verify OpenAPI schema includes all tool paths
2. Check that schema is imported correctly in Actions
3. Test tools directly via MCP protocol
4. Restart ChatGPT conversation to refresh tool definitions

### ngrok URL keeps changing

**Problem:** Free tier ngrok URLs change on restart.

**Solutions:**
1. **Upgrade ngrok:** Paid plans ($8/month) provide static URLs
2. **Update config:** After ngrok restart, update URL in ChatGPT Developer Mode
3. **Use script:** Create a startup script that updates config automatically

---

## Example Conversations

### Indexing a Repository

**You:** "Index my Python project at /home/dave/dev/myapp"

**ChatGPT:** *calls `index_repository` tool*

"I've successfully indexed your repository at /home/dave/dev/myapp. Found:
- 42 files
- 156 functions
- 23 classes
- 89 methods

What would you like to explore?"

### Searching Code

**You:** "Find all functions that handle authentication"

**ChatGPT:** *calls `search_code` tool with query "auth"*

"I found 8 matches for authentication-related code:

1. `auth.py:45` - `def verify_api_key(provided_key: str) -> bool`
2. `auth.py:67` - `def hash_api_key(api_key: str) -> str`
..."

### Finding Definitions

**You:** "Show me where the SigilIndex class is defined"

**ChatGPT:** *calls `goto_definition` tool*

"The `SigilIndex` class is defined in:

**File:** `/home/dave/dev/sigil-mcp-server/indexer.py`
**Line:** 23
**Type:** class

Would you like to see the implementation?"

---

## Advanced Configuration

### IP Whitelisting

Restrict access to specific IPs (useful when exposing via ngrok):

```bash
export SIGIL_MCP_ALLOWED_IPS="203.0.113.1,203.0.113.2"
python server.py
```

### Disabling Authentication (NOT RECOMMENDED)

For local testing only (never use with ngrok):

```bash
export SIGIL_MCP_AUTH_ENABLED=false
python server.py
```

### Custom Index Location

Change where indexes are stored:

```bash
export SIGIL_INDEX_DIR="/path/to/custom/index"
python server.py
```

---

## Additional Resources

- **Sigil Security Guide:** [SECURITY.md](SECURITY.md)
- **MCP Specification:** [https://modelcontextprotocol.io](https://modelcontextprotocol.io)
- **OpenAI MCP Docs:** [https://platform.openai.com/docs/guides/tools-connectors-mcp](https://platform.openai.com/docs/guides/tools-connectors-mcp)
- **OpenAI Actions Docs:** [https://platform.openai.com/docs/actions](https://platform.openai.com/docs/actions)
- **ngrok Documentation:** [https://ngrok.com/docs](https://ngrok.com/docs)

---

## Quick Reference

### Required Headers for API Calls

```http
POST /tools/call HTTP/1.1
Host: your-ngrok-url.ngrok-free.app
Content-Type: application/json
X-API-Key: YOUR_API_KEY_HERE

{
  "name": "search_code",
  "arguments": {
    "query": "function"
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGIL_MCP_AUTH_ENABLED` | `true` | Enable/disable API key auth |
| `SIGIL_MCP_ALLOWED_IPS` | `None` | Comma-separated IP whitelist |
| `SIGIL_REPO_MAP` | `{}` | JSON mapping repo names to paths |
| `SIGIL_INDEX_DIR` | `~/.sigil_index` | Index storage location |

### API Key File Location

```
~/.sigil_mcp_server/api_key
```

Contains the SHA-256 hash of your API key (not the key itself).
