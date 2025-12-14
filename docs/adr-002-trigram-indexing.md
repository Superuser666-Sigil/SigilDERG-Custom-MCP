<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-002: Trigram-Based Text Indexing for Code Search

## Status

Superseded by [ADR-017: rocksdict Trigram Store](adr-017-rocksdb-trigram-store.md). SQLite-backed trigram storage is no longer present in the codebase.

## Context

The Sigil MCP server needs to provide fast, substring-based code search across potentially large repositories. Users expect to search for:

- Function names (exact or partial matches)
- Class names and method names
- Variable names and identifiers
- String literals and comments
- Code patterns (e.g., "async def", "TODO:")

Traditional approaches have limitations:

- **Grep-style search**: Slow for large codebases, requires scanning entire files
- **Full-text search engines** (Elasticsearch, Solr): Heavy dependencies, complex setup
- **Regex engines**: Fast but still require file scanning, no indexing
- **Token-based indexing**: Misses partial matches within identifiers

The solution needs to be fast, lightweight, and work well for code (which contains long identifiers, camelCase, snake_case, etc.).

## Decision

Implement trigram-based inverted indexing inspired by GitHub's Blackbird search engine:

1. **Trigram Generation**: Break all text into overlapping 3-character sequences
2. **Inverted Index**: Map each trigram to documents and positions containing it
3. **Query Processing**: Query strings decomposed into trigrams, intersection of document sets
4. **Storage**: Trigram index persisted via rocksdict (RocksDB bindings)
5. **Hybrid Search**: Combine trigram search with symbol-based search (via ctags)

Example: "async" generates trigrams: "asy", "syn", "ync"

Search algorithm:
```
Query: "search"
Trigrams: ["sea", "ear", "arc", "rch"]
Result: Documents containing ALL four trigrams (intersection)
Post-filter: Verify actual substring exists in candidates
```

## Consequences

### Positive

- **Fast substring search**: O(k) where k = trigrams * documents per trigram, typically 10-100ms
- **Handles partial matches**: Can find "HttpClient" when searching "Client" or "Http"
- **Case-insensitive by default**: Trigrams lowercased during indexing
- **Works with code**: Handles camelCase, snake_case, and long identifiers well
- **Incremental indexing**: Can add/update repositories without full reindex
- **Compression**: Blob storage uses zlib, saves disk space

### Negative

- **Index size**: Trigram indexes can be 2-3x the size of source code
- **Short queries**: Queries under 3 characters fall back to full scan
- **Build time**: Initial indexing takes time (but only done once per repo)
- **False positives**: Trigram intersection can match documents without actual substring (requires post-filtering)
- **Memory usage**: Loading trigram results into memory for intersection

### Neutral

- Trigram index stored in `~/.sigil_index/trigrams.rocksdb`
- Documents and symbols in separate `repos.db` for relational queries
- Compressed content blobs in `blobs/` directory
- SHA-256 deduplication prevents storing identical files multiple times

## Alternatives Considered

### Alternative 1: Regular Expression Search (Grep)

Scan all files using regex engines, no indexing.

**Rejected because:**
- Too slow for large repositories (seconds to minutes)
- No way to avoid scanning every file on every query
- Doesn't scale beyond small projects
- Already implemented as `search_repo` tool for non-indexed searches

### Alternative 2: Full-Text Search Engine (Elasticsearch)

Use Elasticsearch or similar for full-text indexing.

**Rejected because:**
- Heavy dependency (JVM, significant memory usage)
- Complex setup and configuration
- Overkill for local code search
- Poor fit for embedding in Python MCP server
- Would require external service management

### Alternative 3: Token-Based Inverted Index

Index whole tokens (identifiers, keywords) only.

**Rejected because:**
- Misses partial matches (can't find "Client" in "HttpClient")
- Doesn't work well for camelCase or snake_case splitting
- Complex tokenization rules for different languages
- Still requires trigram-like approach for partial matching

### Alternative 4: N-gram with N > 3

Use 4-grams, 5-grams, etc. for indexing.

**Rejected because:**
- Larger N means exponentially more index entries
- Reduces recall (might miss valid matches)
- Trigrams (3-grams) proven effective by Google Code Search, GitHub Blackbird
- 3-character sequences provide good balance of precision and recall

### Alternative 5: Suffix Array / Suffix Tree

Build suffix array or tree data structures for each file.

**Rejected because:**
- Complex implementation for marginal benefit
- Higher memory usage than trigram indexes
- Less intuitive query model
- Harder to persist efficiently in SQLite

## Related

- [Google Code Search paper (2006)](https://swtch.com/~rsc/regexp/regexp4.html)
- GitHub's Blackbird search engine (trigram-based)
- [PostgreSQL pg_trgm module](https://www.postgresql.org/docs/current/pgtrgm.html)
- [Symbol Search ADR](adr-003-symbol-search-ctags.md)
