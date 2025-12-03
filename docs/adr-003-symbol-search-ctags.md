<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-003: Symbol Extraction Using Universal Ctags

## Status

Accepted

## Context

Beyond text search, developers need IDE-like symbol navigation:

- "Go to Definition" for functions, classes, methods
- "Find All References" for symbols
- "View File Structure" showing all symbols in a file
- "List All Symbols" across entire projects

These features require understanding code structure and extracting semantic information:
- Function and method definitions with signatures
- Class declarations and inheritance
- Variable declarations
- Interface definitions
- Type definitions

Building language-specific parsers for Python, JavaScript, Rust, Go, C++, etc. would be:
- Time-consuming (hundreds of languages)
- Error-prone (handling syntax edge cases)
- Maintenance-heavy (keeping up with language changes)

## Decision

Use Universal Ctags for symbol extraction with SQLite storage:

1. **Universal Ctags**: Mature, battle-tested tool supporting 100+ languages
2. **JSON Output**: Parse ctags JSON format for structured symbol data
3. **Symbol Storage**: Store in SQLite `symbols` table with indexes on name, kind, file
4. **Hybrid Search**: Combine trigram text search with symbol-based lookups
5. **Optional Dependency**: Gracefully degrade if ctags not installed (text search still works)

Symbol schema:
```sql
CREATE TABLE symbols (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER,
    name TEXT,
    kind TEXT,  -- 'function', 'class', 'method', 'variable'
    line INTEGER,
    signature TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents(id)
);
CREATE INDEX idx_symbols_name ON symbols(name);
CREATE INDEX idx_symbols_kind ON symbols(kind);
```

## Consequences

### Positive

- **Multi-language support**: 100+ languages out of the box (Python, JavaScript, Rust, Go, C, C++, Java, etc.)
- **Battle-tested**: ctags used by vim, emacs, and many editors for decades
- **Rich metadata**: Function signatures, class hierarchies, line numbers, scopes
- **Maintained**: Universal Ctags actively developed and updated
- **Fast**: Native code (C) for parsing, very efficient
- **Graceful degradation**: If ctags not installed, trigram search still works

### Negative

- **External dependency**: Requires universal-ctags installation on the system
- **Platform differences**: Installation varies (brew, apt, pacman)
- **Language coverage gaps**: Some esoteric languages not supported
- **Parse errors**: Malformed code may cause ctags to skip files
- **Limited semantic understanding**: Ctags does syntax-level parsing, not full semantic analysis

### Neutral

- Ctags invoked as subprocess with JSON output format
- Symbol extraction happens during indexing, not at query time
- Symbols stored per-file, allowing incremental updates
- Indexed by name and kind for fast lookups

## Alternatives Considered

### Alternative 1: Language Server Protocol (LSP)

Use language servers (pylsp, rust-analyzer, etc.) for symbol extraction.

**Rejected because:**
- Requires separate language server installation per language
- Complex lifecycle management (starting, stopping servers)
- Higher memory usage (servers stay resident)
- Async communication overhead (JSON-RPC)
- Overkill for just symbol extraction
- Configuration complexity per language

### Alternative 2: Tree-sitter

Use tree-sitter parsing library for syntax tree generation.

**Rejected because:**
- Requires Python bindings and grammar files per language
- More complex integration than ctags subprocess
- Smaller language coverage than ctags
- Would need custom query logic for symbol extraction
- Still relatively new compared to ctags maturity

### Alternative 3: Regex-Based Symbol Extraction

Write regex patterns to match function/class definitions.

**Rejected because:**
- Extremely fragile (breaks on edge cases)
- Different regex per language
- Can't handle nested scopes, complex syntax
- No access to signature information
- Maintenance nightmare as languages evolve

### Alternative 4: Python AST for Python-Only

Use Python's `ast` module for Python, ignore other languages.

**Rejected because:**
- Only works for Python (many repos are multi-language)
- Doesn't help with JavaScript, Rust, Go, C++ codebases
- Inconsistent experience across languages
- Would need separate solution for each language anyway

### Alternative 5: Commercial Code Intelligence API

Use services like Sourcegraph API or GitHub Code Search.

**Rejected because:**
- Requires uploading code to third-party service (privacy concerns)
- API costs and rate limits
- Network dependency (can't work offline)
- Defeats purpose of local code indexing
- Lock-in to specific vendor

## Related

- [Universal Ctags GitHub](https://github.com/universal-ctags/ctags)
- [Ctags Documentation](https://docs.ctags.io/)
- [Trigram Indexing ADR](adr-002-trigram-indexing.md)
- [Language Server Protocol](https://microsoft.github.io/language-server-protocol/)
- [Tree-sitter](https://tree-sitter.github.io/)
