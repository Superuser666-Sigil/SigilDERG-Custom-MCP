<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Sigil MCP Server [![Version](https://img.shields.io/badge/version-1.0.1-blue)](CHANGELOG.md) [![Tests](https://img.shields.io/badge/tests-408%20passed%20(41s)-brightgreen)](tests) [![Coverage](https://img.shields.io/badge/coverage-76%25-yellowgreen)](coverage.xml) [![Changelog](https://img.shields.io/badge/changelog-CHANGELOG.md-blue)](CHANGELOG.md)

A Model Context Protocol (MCP) server that provides IDE-like code navigation and search for local repositories. Gives AI assistants like ChatGPT powerful code exploration capabilities including symbol search, trigram indexing, and semantic navigation.

## Quickstart

See `docs/QUICKSTART.md` for the fastest path to a working config and the most common knobs (index path, repos, embeddings on/off, admin settings).

## Features

**Hybrid Code Search**
- Fast text search using trigram indexing (inspired by GitHub's Blackbird)
- Trigram store uses RocksDB via `rocksdict` (install with `pip install -e .[trigrams-rocksdict]`); SQLite fallback is removed.
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
pip install -e .[server-full]
```

Default embedding runtime: `llamacpp` with Jina v2 code embeddings (768-dim) at `./models/jina/jina-embeddings-v2-base-code-Q4_K_M.gguf`.

Install Universal Ctags for symbol extraction (optional but recommended):

**macOS:** `brew install universal-ctags`
**Ubuntu/Debian:** `sudo apt install universal-ctags`
**Arch Linux:** `sudo pacman -S ctags`

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
├── repos.db           # SQLite: repos, documents, symbols
├── trigrams.rocksdb/  # RocksDB trigram inverted index (default, via rocksdict)
├── lancedb/       # LanceDB vector store (per-repo code_vectors tables + PQ indexes)
└── blobs/         # Compressed content
```

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

## Documentation

**Setup Guides**
- [ChatGPT Setup Guide](docs/CHATGPT_SETUP.md)
- [OAuth Configuration](docs/OAUTH_SETUP.md)
- [Cloudflare Tunnel Deployment](docs/CLOUDFLARE_TUNNEL.md) 
- [Security Best Practices](docs/SECURITY.md)
- [Operations Runbook](docs/RUNBOOK.md) 
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md) 
- [Llama.cpp Local Embeddings](docs/LLAMACPP_SETUP.md) 
- [Vector Embeddings Usage Guide](docs/VECTOR_EMBEDDINGS.md)
- [Embedding Setup Guide](docs/EMBEDDING_SETUP.md)

**Architecture Decision Records (ADRs)**
- [ADR-001: OAuth 2.0 Authentication](docs/adr-001-oauth2-authentication.md)
- [ADR-002: Trigram-Based Indexing](docs/adr-002-trigram-indexing.md) (superseded)
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
- [ADR-013: LanceDB Vector Store Migration](docs/adr-013-lancedb-vector-store.md)
- [ADR-014: Admin UI Testing Strategy](docs/adr-014-admin-ui-testing.md)
- [ADR-015: Default Llama.cpp + Jina Embeddings](docs/adr-015-default-llamacpp-jina.md)
- [ADR-016: External MCP Aggregation](docs/adr-016-external-mcp-aggregation.md)
- [ADR-017: RocksDB Trigram Store](docs/adr-017-rocksdb-trigram-store.md)

**Other**
- [ChatGPT OAuth Configuration](docs/CHATGPT_OAUTH_CONFIG.md)
- [Cloudflare 502 Fix](docs/CLOUDFLARE_502_FIX.md)
- [Cloudflare OAuth Issue](docs/CLOUDFLARE_OAUTH_ISSUE.md)
- [External MCP Integration](docs/external_mcp.md) (if exists, otherwise see config.example.json)

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
