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
- Added `rebuild_indexes.py` maintenance script to completely wipe and rebuild all indexes (documents, symbols, trigrams, and embeddings) across all configured repositories.

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
