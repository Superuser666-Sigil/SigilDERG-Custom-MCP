<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Sigil MCP Server

A Model Context Protocol (MCP) server that provides IDE-like code navigation and search for local repositories. Gives AI assistants like ChatGPT powerful code exploration capabilities including symbol search, trigram indexing, and semantic navigation.

## Features

**Hybrid Code Search**
- Fast text search using trigram indexing (inspired by GitHub's Blackbird)
- Symbol-based search for functions, classes, methods, and variables
- Semantic code search with vector embeddings backed by LanceDB (ANN queries, per-repo vector stores)
- File structure view showing code outlines
- Automatic index updates with file watching (optional)

**Production Ready**
- Thread-safe concurrent access (SQLite WAL mode + RLock serialization)
- File watcher, HTTP handlers, and vector indexing run safely in parallel
- No "database is locked" errors from concurrent operations
- Admin API for operational management (index rebuilds, stats, logs)
- Comprehensive request/response logging with header redaction

**Enterprise Security**
- OAuth 2.0 authentication with PKCE support for remote access
- Local connection bypass (no auth needed for localhost)
- API key fallback and IP whitelisting

**Available Tools**
- `index_repository` - Build searchable index with symbol extraction
- `search_code` - Fast substring search across repositories
- `goto_definition` - Find symbol definitions
- `list_symbols` - View file/repo structure
- `list_mcp_tools`, `external_mcp_prompt` - Discover external MCP tools registered into Sigil
- `build_vector_index` - Generate semantic embeddings for code (optional)
- `semantic_search` - Natural language code search using embeddings
- `list_repos`, `read_repo_file`, `list_repo_files`, `search_repo` - Basic operations
- `get_index_stats`, `ping` - Server info and health checks

## Quick Start

### Installation

Clone and install dependencies:

```bash
git clone https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP.git
cd SigilDERG-Custom-MCP
pip install -e .

# Optional: Install file watching support
pip install -e .[watch]

# Optional: Install vector embeddings - choose based on your hardware:

# For sentence-transformers (NVIDIA GPUs, or CPU)
pip install -e .[embeddings-sentencetransformers]

# For OpenAI API (cloud-based)
pip install -e .[embeddings-openai]

# For llama.cpp - choose your acceleration:
pip install -e .[embeddings-llamacpp-cpu]      # CPU only
pip install -e .[embeddings-llamacpp-cuda]     # NVIDIA GPU (CUDA)
pip install -e .[embeddings-llamacpp-rocm]     # AMD GPU (ROCm)
pip install -e .[embeddings-llamacpp-metal]    # Apple Silicon (Metal)

# Or install all embedding providers (not recommended)
pip install -e .[embeddings-all]

# Everything for the default stack (Llama.cpp + LanceDB + watcher)
pip install -e .[server-full]
```

Default embedding runtime: `llamacpp` with Jina v2 code embeddings (768-dim) at `/home/dave/models/jina/jina-embeddings-v2-base-code-Q4_K_M.gguf`.
Ensure the model file exists, or override `embeddings.provider`/`embeddings.model` in `config.json` if you prefer a different backend.

Install Universal Ctags for symbol extraction (optional but recommended):

**macOS:** `brew install universal-ctags`
**Ubuntu/Debian:** `sudo apt install universal-ctags`
**Arch Linux:** `sudo pacman -S ctags`

### Default stack (Llama.cpp + Jina + LanceDB)

1. Install dependencies: `pip install -e .[server-full]`.
2. Download the Jina code embedding model (e.g., `jina-embeddings-v2-base-code-Q4_K_M.gguf`) into `./models/` (or set `SIGIL_MCP_MODELS`/`embeddings.model` to your path).
3. Enable embeddings in `config.json`:

```json
{
  "mode": "dev",
  "embeddings": {
    "enabled": true,
    "provider": "llamacpp",
    "model": "./models/jina-embeddings-v2-base-code-Q4_K_M.gguf"
  },
  "authentication": {
    "enabled": false,
    "allow_local_bypass": true
  }
}
```

4. Start the server: `SIGIL_MCP_MODE=dev sigil-mcp`. Check `/readyz` for a 200 once indexes and models are ready.

When embeddings are disabled or LanceDB is missing, the server falls back to trigram search and logs the reason so you can fix it later.

### Dev vs Prod

Set `SIGIL_MCP_MODE` (or `mode` in `config.json`) to switch security defaults:

- `dev` (default): authentication disabled, local bypass allowed, admin API key optional.
- `prod`: authentication enabled, local bypass disabled, admin API requires an API key and IP whitelist. Insecure overrides log warnings at startup.

Recommended production snippet:

```json
{
  "mode": "prod",
  "authentication": {
    "enabled": true,
    "allow_local_bypass": false
  },
  "admin": {
    "enabled": true,
    "require_api_key": true,
    "api_key": "generate-and-set-this",
    "allowed_ips": ["127.0.0.1"]
  }
}
```

### Admin API and readiness

- `/readyz` returns 200 only when config is loaded, indexes are open, and embeddings are initialized (when enabled); otherwise 503.
- In production, the Admin API returns 503 unless `admin.api_key` is set and the request comes from an allowed IP. CORS is limited to known admin UI hosts (no wildcards).

### Hardware guidance

- CPU-only works for trigram search and small Llama.cpp models; vector indexing benefits from fast SSDs and >8GB RAM.
- GPU (CUDA/ROCm/Metal) improves embedding throughput; install the matching `embeddings-llamacpp-*` extra.

### Contributing & licensing

Contributions are welcome, but submitting code means agreeing to the CLA in `CLA.md` and licensing changes under AGPLv3. See `CONTRIBUTING.md` for the full process and links.

### Configuration

Copy the example config and edit with your repository paths:

```bash
cp config.example.json config.json
# Edit config.json
```

Example configuration:

```json
{
  "repositories": {
    "my_project": "/absolute/path/to/your/project",
    "another_repo": "/path/to/another/repo"
  }
}
```

Alternatively, use environment variables:

```bash
export SIGIL_REPO_MAP="my_project:/path/to/project;another:/path/to/another"
```

### External MCP servers (Playwright, Next.js MCP, MindsDB, etc.)

Sigil can aggregate external MCP servers and expose their tools with a server prefix (e.g., `playwright.click`, `mindsdb.query`).

1. Add entries to `config.json` under `external_mcp_servers` (see `config.example.json` and `docs/external_mcp.md` for samples and auth header guidance). Use headers for tokens: `"authorization": "Bearer <token>"`.
2. Optional: set `external_mcp_auto_install` to `true` to run `npx`/`npm`/`bunx` commands defined on startup (disabled by default).
3. On startup, Sigil registers external tools automatically. Check status via `GET /admin/mcp/status` or call the `list_mcp_tools`/`external_mcp_prompt` tools from your MCP client.
4. Client preset: `docs/mcp.json` includes Sigil + sample Playwright/Next.js MCP/MindsDB entries—update URLs/tokens to match your environment.

### Running the Server

**Recommended: Use the restart script (starts both MCP server and Admin UI):**

```bash
./scripts/restart_servers.sh
```

This script will:
- Stop any running server processes
- Start the MCP Server on port 8000
- Start the Admin UI frontend on port 5173
- Run both processes with `nohup` so they persist after terminal closes

**Manual start (MCP server only):**

```bash
python -m sigil_mcp.server
```

**Stop all servers:**

```bash
./scripts/restart_servers.sh --stop
```

On first run, OAuth credentials will be generated. Save the Client ID and Client Secret for connecting from ChatGPT.

### Connecting to ChatGPT

> [!IMPORTANT]
> **Using Cloudflare Tunnel?** You must disable Bot Fight Mode or ChatGPT's OAuth will fail.  
> 📖 See [**Cloudflare OAuth Issue & Solution**](docs/CLOUDFLARE_OAUTH_ISSUE.md) for details.

1. Expose via ngrok: `ngrok http 8000` (or use Cloudflare Tunnel)
2. In ChatGPT, add MCP connector with OAuth authentication
3. Use the OAuth credentials from server startup
4. Start using: "Search my code for async functions"

**Important**: The server is configured for ChatGPT compatibility:
- DNS rebinding protection is disabled (ChatGPT sends ngrok Host headers)
- MCP endpoint mounted at root `/` (not `/mcp`)
- OAuth authentication remains active and required

See [docs/CHATGPT_SETUP.md](docs/CHATGPT_SETUP.md) for detailed instructions.

## Configuration

### Using config.json

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
    "repo_name": "/absolute/path/to/repo"
  },
  "watch": {
    "enabled": true,
    "debounce_seconds": 2.0,
    "ignore_dirs": [".git", "__pycache__", "node_modules", "build"],
    "ignore_extensions": [".pyc", ".so", ".pdf", ".png"]
  },
  "index": {
    "path": "~/.sigil_index"
  },
  "admin": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 8765,
    "api_key": null,
    "allowed_ips": ["127.0.0.1", "::1"]
  }
}
```

### Using Environment Variables

```bash
export SIGIL_MCP_HOST=127.0.0.1
export SIGIL_MCP_PORT=8000
export SIGIL_MCP_AUTH_ENABLED=true
export SIGIL_MCP_OAUTH_ENABLED=true
export SIGIL_MCP_ALLOW_LOCAL_BYPASS=true
export SIGIL_MCP_WATCH_ENABLED=true
export SIGIL_MCP_WATCH_DEBOUNCE=2.0
export SIGIL_REPO_MAP="name1:/path/to/repo1;name2:/path/to/repo2"
export SIGIL_INDEX_PATH=~/.sigil_index
```

### File Watching (Optional)

Enable automatic index updates when files change:

```bash
# Install watchdog
pip install .[watch]

# Enable in config.json or via environment
export SIGIL_MCP_WATCH_ENABLED=true
```

The server will:
- **Granularly re-index** individual files as they change (modified/created)
- **Batch updates** with configurable debounce (default 2 seconds)
- **Smart filtering** using configurable ignore patterns

Configure what to ignore in `config.json`:

```json
{
  "watch": {
    "enabled": true,
    "debounce_seconds": 2.0,
    "ignore_dirs": [".git", "__pycache__", "coverage", "htmlcov"],
    "ignore_extensions": [".pyc", ".so", ".pdf", ".png", ".jpg"]
  }
}
```

Environment variables:
```bash
export SIGIL_MCP_WATCH_ENABLED=true
export SIGIL_MCP_WATCH_DEBOUNCE=2.0
```

## Authentication

**OAuth 2.0 (Recommended for Remote Access)**

OAuth credentials are generated on first run. Supports PKCE for enhanced security and token-based authentication with refresh capabilities. See [docs/OAUTH_SETUP.md](docs/OAUTH_SETUP.md) for details.

**Local Development**

Localhost connections automatically bypass authentication. No credentials needed when connecting from 127.0.0.1.

**API Key Fallback**

```bash
export SIGIL_MCP_API_KEY=your_secure_key_here
```

See [docs/SECURITY.md](docs/SECURITY.md) for security best practices.

## Admin API & Admin UI

The Admin API provides HTTP endpoints for operational management without restarting the server. It's **integrated into the main MCP server** (port 8000) and accessible at `/admin/*` endpoints. A React-based Admin UI is available on port 5173.

**Start everything (recommended):**
```bash
./scripts/restart_servers.sh
```

This starts both:
- MCP Server on `http://127.0.0.1:8000` (includes Admin API at `/admin/*`)
- Admin UI on `http://localhost:5173`

**Admin API Endpoints:**
- `GET /admin/status` - Server status, repositories, index info
- `POST /admin/index/rebuild` - Rebuild index (all repos or specific)
- `GET /admin/index/stats` - Get index statistics
- `POST /admin/vector/rebuild` - Rebuild vector embeddings
- `GET /admin/logs/tail` - View recent log entries
- `GET /admin/config` - View current configuration

**Security:**
- IP whitelist (default: localhost only)
- Optional API key authentication

**Manual start (Admin UI only, for development):**
```bash
cd sigil-admin-ui
npm install
npm run dev
```

Then open `http://localhost:5173` in your browser.

See [docs/RUNBOOK.md](docs/RUNBOOK.md#admin-api) for complete Admin API documentation, including backup/migration guidance for the LanceDB vector store.

### Admin UI Testing & Coverage

The React Admin UI ships with a Vitest + Testing Library suite that exercises the Status, Index, Vector, Logs, and Config flows end-to-end (dialog interactions, auto-refresh timers, clipboard handling, error states, etc.). Before merging UI changes:

- Run `npm run test -- --coverage` inside `sigil-admin-ui`.
- Keep overall coverage ≥70% and critical pages/components (Status/Index/Vector/Logs/Config plus shared API utilities) at 100% line coverage.
- When adding components, include deterministic timers (use `setInterval` guards) and `data-testid` hooks where necessary to avoid brittle queries.

Refer to [docs/adr-014-admin-ui-testing.md](docs/adr-014-admin-ui-testing.md) for the rationale and guardrails.

## Usage Examples

Once connected to ChatGPT as an MCP server:
```
You: "Index my project repository"
ChatGPT: Indexed 342 files, found 1,847 symbols in 3.2 seconds

You: "Find where the HttpClient class is defined"
ChatGPT: Found in project::src/http/client.py at line 45

You: "Search for async functions"
ChatGPT: Found 23 matches across 8 files

You: "Build vector index for semantic search"
ChatGPT: Indexed 856 chunks from 342 documents

You: "Find code that handles user authentication"
ChatGPT: Found 5 relevant code sections (semantic search):
  - auth/handlers.py:45-145 (score: 0.89)
  - middleware/auth.py:12-112 (score: 0.84)
  ...
```tGPT: Found 23 matches across 8 files
```

## Architecture

**Indexing Process**

1. File scanning (skips build artifacts)
2. Content storage with SHA-256 deduplication
3. Symbol extraction via universal-ctags
4. Trigram inverted index generation
5. Compression using zlib

**Storage**
```
~/.sigil_index/
├── repos.db       # SQLite: repos, documents, symbols
├── trigrams.db    # SQLite: trigram inverted index
├── lancedb/       # LanceDB vector store (per-repo code_vectors tables + PQ indexes)
└── blobs/         # Compressed content
``` blobs/         # Compressed content
```

> 📦 **Backups:** Include the `lancedb/` directory when snapshotting your index path. The SQLite databases no longer contain embeddings after the LanceDB migration described in [ADR-013](docs/adr-013-lancedb-vector-store.md).

**Performance**

- Symbol lookup: O(log n) via SQLite indexes
- Text search: O(k) where k = trigrams * documents per trigram
- Typical query latency: 10-100ms

## Security

**Path Traversal Protection:** All paths validated to prevent escaping repository roots

**Authentication Layers:** OAuth 2.0 (primary), Local bypass (localhost), API keys (fallback), IP whitelist (optional)

**Protection:** Source code requires authentication for remote access, OAuth credentials stored with 0600 permissions, tokens expire after 1 hour with refresh support, PKCE prevents authorization code interception

**ChatGPT Compatibility**: For ChatGPT MCP connector compatibility, DNS rebinding protection is disabled. This means:
- [NO] Host header validation: Disabled (accepts ngrok domains)
- [NO] Content-Type validation: Disabled (accepts application/octet-stream)
- [YES] OAuth 2.0 authentication: Active and required
- [YES] Bearer token validation: Active
- [YES] Token expiration: Enforced

See [docs/SECURITY.md](docs/SECURITY.md) for detailed security documentation.

## Troubleshooting

For detailed troubleshooting, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) and [docs/RUNBOOK.md](docs/RUNBOOK.md).

**Quick fixes:**

**"ctags not available":** Install universal-ctags (see Quick Start). Text search works without it.

**"No repositories configured":** Set repositories in config.json or SIGIL_REPO_MAP environment variable.

**"Authentication failed":** For localhost, verify allow_local_bypass is true. For remote, verify OAuth credentials.

**"watchdog not available":** Install with `pip install sigil-mcp-server[watch]` to enable file watching.

**More help:** See comprehensive [Troubleshooting Guide](docs/TROUBLESHOOTING.md) and [Operations Runbook](docs/RUNBOOK.md).

## Request Logging

All HTTP requests and responses are automatically logged with:
- Request headers (sensitive data redacted)
- Response status codes and duration
- Client IP addresses
- Cloudflare ray IDs (if using Cloudflare)
- Request IDs for correlation

This provides full visibility for debugging client issues and support tickets. Sensitive headers (authorization, cookies, API keys) are automatically redacted before logging.

See [docs/adr-012-header-logging-middleware.md](docs/adr-012-header-logging-middleware.md) for details.

## Documentation

**Setup Guides**
- [ChatGPT Setup Guide](docs/CHATGPT_SETUP.md)
- [OAuth Configuration](docs/OAUTH_SETUP.md)
- [Cloudflare Tunnel Deployment](docs/CLOUDFLARE_TUNNEL.md) 
- [Security Best Practices](docs/SECURITY.md)
- [Operations Runbook](docs/RUNBOOK.md) 
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md) 
- [Llama.cpp Local Embeddings](docs/LLAMACPP_SETUP.md) 

**Architecture Decision Records (ADRs)**
- [ADR-001: OAuth 2.0 Authentication](docs/adr-001-oauth2-authentication.md)
- [ADR-002: Trigram-Based Indexing](docs/adr-002-trigram-indexing.md)
- [ADR-003: Symbol Extraction with Ctags](docs/adr-003-symbol-search-ctags.md)
- [ADR-004: JSON Configuration System](docs/adr-004-configuration-system.md)
- [ADR-005: FastMCP Custom Routes](docs/adr-005-fastmcp-custom-routes.md)
- [ADR-006: Vector Embeddings for Semantic Search](docs/adr-006-vector-embeddings.md)
- [ADR-007: File Watching](docs/adr-007-file-watching.md)
- [ADR-008: Granular Re-indexing and Configurable Patterns](docs/adr-008-granular-indexing.md)
- [ADR-009: ChatGPT MCP Connector Compatibility](docs/adr-009-chatgpt-compatibility.md)
- [ADR-010: Thread Safety and SQLite WAL Mode](docs/adr-010-thread-safety-sqlite.md)
- [ADR-011: Admin API for Operational Management](docs/adr-011-admin-api.md)
- [ADR-012: ASGI Header Logging Middleware](docs/adr-012-header-logging-middleware.md)

**Feature Documentation**
- [Vector Embeddings Usage Guide](docs/VECTOR_EMBEDDINGS.md)
- [Llama.cpp Setup Guide](docs/LLAMACPP_SETUP.md)

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines including:
- [Contributor License Agreement (CLA)](CLA.md) - **Required for all contributors**
- Developer Certificate of Origin (DCO) requirements
- Code standards and testing requirements
- Pull request process
- Code of Conduct

## Licensing

Sigil is dual-licensed:

- **Open Source**: Available under AGPLv3 for open-source projects and private use where source sharing requirements are met.

- **Commercial**: A commercial license is required for organizations who wish to run Sigil internally without open-sourcing their own applications or who need indemnification and support.

[Contact me](mailto:davetmire85@gmail.com) for commercial licensing options.

See [LICENSE](LICENSE) file for full AGPLv3 text.

### Licensing FAQ

**Q: Can I run this inside my company under AGPLv3?**

A: Yes, as long as you're comfortable with AGPLv3 and its requirements. If you expose the server to users over a network (like running it as an internal service), AGPLv3 requires making the source code available to those users, including any modifications you've made.

**Q: We have a "no AGPL" policy. Can we still use Sigil?**

A: Yes, via a commercial license. Email [davetmire85@gmail.com](mailto:davetmire85@gmail.com) to discuss your needs.

**Q: Why do I have to sign a CLA to contribute?**

A: The Contributor License Agreement keeps the licensing story clean—AGPLv3 for the open-source community, commercial licenses for organizations that need them—without legal ambiguity about who owns what. Your contribution remains open-source under AGPLv3; the CLA just clarifies the rights.

**Q: What's included in a commercial license?**

A: Commercial licenses provide freedom to use Sigil internally without open-source requirements, ability to keep modifications proprietary, indemnification and support options, and clear legal status for enterprise compliance. Contact me for details and pricing.

**Q: Can I use this for my personal projects?**

A: Absolutely! AGPLv3 is perfect for personal projects, hobbyist use, and small teams. You only need a commercial license if you have organizational requirements that conflict with AGPL.

For more details on contributing, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments

- Trigram indexing inspired by GitHub's Blackbird search engine
- Symbol extraction powered by Universal Ctags
- Built on the Model Context Protocol (MCP) specification

## Support

Issues: [GitHub Issues](https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP/issues)
Documentation: [docs/](docs/)
Security: [docs/SECURITY.md](docs/SECURITY.md)
