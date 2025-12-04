# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/yourusername/sigil-mcp-server/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/yourusername/sigil-mcp-server/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/yourusername/sigil-mcp-server/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yourusername/sigil-mcp-server/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/yourusername/sigil-mcp-server/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/yourusername/sigil-mcp-server/releases/tag/v0.1.0
