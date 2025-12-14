<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-12-14

### Added
- Semantic search rerank controls: `code_only` (hard filter to code) and `prefer_code` (boost code while allowing docs/config), with metadata-aware reranking over a larger candidate pool.
- Metadata tagging for vector rows (`is_code`, `is_doc`, `is_config`, `is_data`, `extension`, `language`) plus lightweight extension/content heuristics to keep language authoritative only for code.
- Llama.cpp config mapping for larger contexts: `llamacpp_context_size` / `n_ctx` now map to provider `context_size`.
- Tests covering code-only/prefer-code rerank behavior and LanceDB stubs supporting the updated search flow.

### Changed
- Trigram backend is rocksdict-only; SQLite/RocksDB fallbacks removed and docs/ADRs updated accordingly.
- Postings are fixed-width encoded (uint32/uint64), intersection ordering is deterministic (shortest-first) with early exits for faster queries.
- Semantic search scoring applies metadata penalties/boosts in a rerank step (preserving vector recall) instead of altering distances.
- Config persistence now prefers the originally loaded path, preventing test/admin writes from overwriting project config; default index path is `~/.sigil_index`.
- Documentation refreshed: rocksdict-only backend, llama.cpp context-size guidance, metadata/rerank behavior, and legacy SQLite embeddings cleanup marked as legacy-only.

### Fixed
- LanceDB in-memory stubs now mirror the real API enough to support the new search flow (including distance propagation).
- Admin rebuild/logs flows benefit from stable config path handling and correct storage location defaults.

## [0.9.0] - 2025-12-11

### Added
- Comprehensive coverage expansion for admin API/UI, app factory, embeddings, LanceDB integration, OAuth flows, auth/security, MCP client/installer, and scripts; added in-memory LanceDB stub for fast, deterministic testing (opt-in via `SIGIL_MCP_LANCEDB_STUB`).
- Test fixtures now isolate OAuth storage via `SIGIL_MCP_OAUTH_DIR` to avoid permission issues in CI.
- New admin API/regression suites for rebuild/status/logs, MCP client manager tests, rebuild-index script tests, and logging setup coverage.

### Changed
- Config defaults now treat embeddings as disabled unless explicitly set; `get_config` refreshes when env changes during tests to keep isolation safe.
- OAuth module supports configurable storage root (`SIGIL_MCP_OAUTH_DIR`) for secure, test-friendly credentials handling.
- File watcher/indexer cleanup hardened to remove LanceDB rows and persist vectors across reopen; LanceDB stub registry preserves state between instances in tests.
- Packaging: PyPI build verified; temp build artifacts removed from package discovery.

### Fixed
- Ready/admin tests now pass with production gating and embeddings toggles; index removal clears vectors and trigrams reliably.
- PyPI build now succeeds after cleaning temporary index dirs and installing `build` tooling.

## [0.8.0] - 2025-12-11

### Added
- External MCP aggregation with server-prefixed tools (`external_mcp_servers` config, env override `SIGIL_MCP_SERVERS`). Diagnostics tools `list_mcp_tools` and `external_mcp_prompt` expose discovered tools.
- Admin MCP endpoints: `GET /admin/mcp/status` and `POST /admin/mcp/refresh` for external MCP visibility and re-discovery.
- Optional auto-install for MCP servers (`external_mcp_auto_install` and per-server `auto_install`) to run `npx`/`npm`/`bunx` commands on startup (disabled by default).
- SSE MCP transport path with optional bearer gating; configurable via `mcp_server` section.
- Admin UI autostart integrated with the main server (configurable command/args/path/port).
- Agent presets and docs for external MCP usage (`docs/AGENTS.md`, `docs/CLAUDE.md`, `docs/mcp.json`, `docs/external_mcp.md`).
- ADR-016 documenting external MCP aggregation decisions.

### Changed
- README and RUNBOOK updated with external MCP setup, admin MCP endpoints, auto-install option, and client presets.
- `config.example.json` expanded with external MCP samples, auto-install flag, admin UI autostart settings, and SSE/bearer transport fields.
- Test suites extended for external MCP config parsing, transport routing, client registration, and installer behavior.

## [0.7.0] - 2025-12-10

### Added
- Deployment `mode` (dev/prod) with secure defaults, environment override, and startup warnings when production is misconfigured.
- Readiness endpoint `/readyz` that reports component readiness (config, indexes, embeddings) for orchestrators.
- Admin API hardening in production: API key required, IP whitelist enforced, and CORS restricted to known Admin UI origins.
- Optional dependency groups for LanceDB (`.[lancedb]`) and a bundled default stack (`.[server-full]`); Dockerfile installs the default extras.
- Tests covering mode defaults, readiness, admin API gating, LanceDB handling, and authentication IP whitelist behavior.

### Changed
- Authentication now enforces IP whitelist checks before local bypass, and authentication failures log explicit reasons.
- Embedding providers emit clearer install/model error messages and disable embeddings gracefully when a backend/model is missing; `llama.cpp` embeddings now chunk large documents before embedding.
- Indexer guards LanceDB imports, falls back to trigram search when embeddings/LanceDB are unavailable, and logs vector index status (path and chunk counts) at startup/rebuild.
- Admin API endpoints return structured errors for locked or misconfigured vector builds; admin endpoints now run inline with the main server process and share a single index instance.
- Runbook/README updated with dev vs prod guidance, default stack install, and readiness/admin security notes; Dockerfile exposes a model volume mount.

### Fixed
- Vector rebuild path no longer raises `config` scoping errors (affecting Admin UI vector page).
- Duplicate MCP tool registration warnings removed by centralizing tool registration.

## [0.6.0] - 2025-12-09

### Added
- ADR-015 documenting the new default embedding backend (llama.cpp + Jina v2 GGUF) and operational requirements.
- `scripts/restart_servers.sh` - Main entrypoint script for starting/stopping all server processes
  - Automatically stops any running MCP Server or Frontend processes
  - Starts MCP Server with `nohup` (port 8000, logs to `/tmp/sigil_server.log`)
  - Starts Admin UI frontend with `nohup` (port 5173, logs to `/tmp/frontend.log`)
  - Verifies both servers started successfully
  - Processes persist after terminal closes
  - Use `./scripts/restart_servers.sh --stop` to stop all servers

### Changed
- Default embeddings now use llama.cpp with the Jina embeddings v2 base code GGUF at `/home/dave/models/jina/jina-embeddings-v2-base-code-Q4_K_M.gguf`
  and a 768-dimension LanceDB schema; docs updated with setup and rebuild steps.
- Embeddings remain LanceDB-backed at `index_dir/lancedb/`; legacy SQLite `embeddings` table remains removed.
- **Admin API Integration**: Admin API now runs integrated into the main MCP server process (port 8000, `/admin/*` endpoints)
  - No longer requires separate service or port
  - Shares the same index instance (eliminates database lock conflicts)
  - `admin_api_main.py` is deprecated (Admin API starts automatically with main server)
- **Server Startup**: `scripts/restart_servers.sh` is now the recommended way to start all servers
  - Updated README.md, RUNBOOK.md, and all documentation to reference the restart script
  - All server processes run with `nohup` for persistence

## [0.4.0] - 2025-12-04

### Added
- Added Admin API (`sigil_mcp/admin_api.py`) - Separate HTTP service for operational management
  - `GET /admin/status` - Server status, repositories, index info, watcher status
  - `POST /admin/index/rebuild` - Rebuild trigram/symbol index (all repos or specific repo)
  - `GET /admin/index/stats` - Get index statistics (all repos or specific repo)
  - `POST /admin/vector/rebuild` - Rebuild vector embeddings index
  - `GET /admin/logs/tail` - Get last N lines from server log file
  - `GET /admin/config` - View current configuration (read-only)
- Added Admin API configuration properties to `Config` class:
  - `admin.enabled` - Enable/disable Admin API (default: true)
  - `admin.host` - Admin API host (default: 127.0.0.1)
  - `admin.port` - Admin API port (default: 8765)
  - `admin.api_key` - Optional API key for additional security
  - `admin.allowed_ips` - IP whitelist (default: 127.0.0.1, ::1)
- Added `HeaderLoggingASGIMiddleware` - ASGI middleware for comprehensive request/response logging
  - Logs all HTTP requests with redacted headers
  - Logs response status codes and duration
  - Extracts client IP from X-Forwarded-For or direct connection
  - Extracts Cloudflare ray IDs for correlation
  - Generates request IDs for request/response correlation
  - Redacts sensitive headers (authorization, cookies, API keys)
- Added operational helper functions to `server.py`:
  - `rebuild_index_op()` - Rebuild trigram/symbol index (used by Admin API and MCP tools)
  - `build_vector_index_op()` - Rebuild vector embeddings index
  - `get_index_stats_op()` - Get index statistics
- Added `test_header_logging.py` - Comprehensive test suite for header logging middleware
- Added `test_admin_api.py` - Test suite for Admin API endpoints

### Changed
- Improved config loading behavior: explicit non-existent paths now skip file-based loading and fall back to environment variables
- Reduced OAuth route header logging: OAuth routes now rely on middleware for header logging (with redaction) instead of logging raw headers
- Enhanced error handling in `semantic_search`: gracefully handles missing documents (deleted files with orphaned embeddings)

### Fixed
- Fixed type annotation issue in `get_index_stats` MCP tool (return type now matches implementation)
- Fixed potential `None` subscript error in `semantic_search` when document is deleted but embedding still exists

## [0.3.2] - 2025-12-04

### Added
- Added `test_client.py` script for end-to-end testing of indexing, search, and embeddings from the command line.
- Added `scripts/rebuild_indexes.py` maintenance script to completely wipe and rebuild all indexes (documents, symbols, trigrams, and embeddings) across all configured repositories.

### Changed
- Improved CONTRIBUTING.md with clearer dual-licensing strategy
  - Added upfront "who this is for" statement
  - Clarified that CLA enables dual-licensing model
  - Explained DCO/CLA relationship
  - Fixed branching model (main branch only)
  - Added "For Companies" section
- Enhanced CODE_OF_CONDUCT.md with explicit scope clarification
- Added comprehensive Licensing FAQ to README.md covering company use, AGPL policies, CLA rationale, and commercial licensing

### Fixed
- File watcher now fully respects `watch.ignore_dirs` and `watch.ignore_extensions` from configuration when deciding which files to index, replacing previous hardcoded ignore sets and improving alignment with ADR-007/ADR-008.
- Added `SigilIndex.remove_file` API and wired it into the file watcher so that deleted files are proactively removed from documents, symbols, embeddings, trigrams, and blob storage without requiring a full index rebuild.
- `semantic_search` MCP tool now returns a structured error when embeddings are not configured instead of crashing the server process.

## [0.3.1] - 2025-01-03

### Fixed
- **Critical:** Fixed path handling bugs preventing all file operations
  - Fixed `_get_repo_root()` to convert string paths to Path objects
  - Fixed `list_repo_files()` to ensure Path objects for all operations
  - Fixed `search_repo()` to convert repo_root to Path before rglob
  - Prevented TypeError: "unsupported operand type(s) for /: 'str' and 'str'"
  - Prevented AttributeError: "'str' object has no attribute 'rglob'"
- Root cause: REPOS dictionary stored paths as strings from config.json, but file operations expected Path objects

### Changed
- Updated TROUBLESHOOTING.md with path handling error diagnosis
- Updated CHATGPT_SETUP.md prerequisites for v0.3.1
- Updated RUNBOOK.md with upgrade instructions and version notes

### Tested
- Verified with fresh index rebuild: 1,153 documents, 17,868 symbols across 6 repositories
- All MCP tools confirmed working: list_repo_files, read_repo_file, search_repo, search_code, goto_definition, list_symbols

## [0.3.0] - 2025-01-03

### Added
- Real-time file watching with watchdog library
- Granular indexing: track files, functions, and classes separately
- Background file watcher for automatic index updates
- ADR-007: File watching architecture
- ADR-008: Granular indexing design
- ADR-009: ChatGPT compatibility guide

### Changed
- Optimized indexing to skip unchanged files
- Updated documentation with file watching and granular indexing usage
- Improved test coverage for new features
- Enhanced RUNBOOK.md with detailed operational procedures

## [0.2.0] - 2025-01-03

### Added
- Configuration-based embedding provider selection (sentence-transformers, OpenAI, llama.cpp)
- Pluggable provider architecture with factory pattern
- 6 new config properties: enabled, provider, model, dimension, cache_dir, api_key
- Comprehensive test suite: 18 embedding tests, 100% passing
- File watching test suite: 495 tests

### Fixed
- **Critical:** Embeddings now use real providers instead of random noise stub

### Changed
- Refactored to Python 3.12 best practices with top-level imports and availability flags
- Updated ADR-006 with provider architecture details
- Updated EMBEDDING_SETUP.md with configuration examples
- Updated RUNBOOK.md with embedding provider troubleshooting
- Cleaned up Flake8 warnings in test files
- Removed .flake8 from .gitignore (config files should be tracked)

## [0.1.1] - 2025-01-03

### Added
- Contributor License Agreement (CLA.md) with dual-licensing framework
- DCO (Developer Certificate of Origin) signing requirement

## [0.1.0] - 2025-01-03

### Added
- Initial release with AGPLv3 license
- MCP server implementation with FastMCP framework
- OAuth 2.0 authentication system
- Trigram-based code search indexing
- SQLite-based repository and symbol database
- Code navigation tools: list_repo_files, read_repo_file, search_repo, search_code
- Symbol search with ctags integration: goto_definition, list_symbols
- Vector embeddings support (sentence-transformers, OpenAI, llama.cpp)
- Comprehensive documentation suite:
  - CHATGPT_SETUP.md: Integration guide
  - EMBEDDING_SETUP.md: Embedding provider configuration
  - LLAMACPP_SETUP.md: Local LLM setup
  - OAUTH_SETUP.md: Authentication configuration
  - RUNBOOK.md: Operational procedures
  - SECURITY.md: Security policies
  - TROUBLESHOOTING.md: Common issues and solutions
  - VECTOR_EMBEDDINGS.md: Embeddings architecture
- Architecture Decision Records (ADR 001-008)
- Test suite with 100% passing tests
- Contributor guidelines (CONTRIBUTING.md)
- Code of Conduct (Contributor Covenant 2.1)

[Unreleased]: https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP/releases/tag/v0.1.0
