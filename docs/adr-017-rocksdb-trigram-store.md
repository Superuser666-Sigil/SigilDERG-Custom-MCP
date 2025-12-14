<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-017: rocksdict Trigram Store (Supersedes ADR-002)

## Status

Accepted — Supersedes [ADR-002: Trigram-Based Text Indexing](adr-002-trigram-indexing.md). SQLite trigram backends and RocksDB references have been removed from the codebase; rocksdict is the only supported postings store.

## Context

The original trigram index (ADR-002) stored postings in SQLite. Under higher write concurrency (file watcher + admin rebuilds + embedding jobs), SQLite’s coarse page locks produced `database is locked` errors and long write latencies. We need a write-friendly, embeddable store that:

- Supports frequent small writes without long transactions.
- Keeps read performance fast for trigram lookups.
- Works fully on-prem with minimal operational burden.
- Scales better for customers with larger codebases and heavier write loads.

## Decision

Adopt rocksdict (RocksDB bindings) as the **only** backend for trigram postings. SQLite trigram support has been removed to simplify the codebase and eliminate fallback complexity.

## Rationale

- **Write-friendliness:** rocksdict’s RocksDB-backed LSM tree and memtables handle high write rates without long writer locks, reducing contention compared to SQLite B-trees.
- **Concurrency:** Readers are effectively lock-free and writers are short-lived; WAL and compaction happen in the background.
- **On-prem & lightweight:** Runs in-process, no external service to operate, and compatible with our existing deployment model.
- **Headroom:** Even if current repos are modest, RocksDB provides room for larger customer codebases and heavier concurrent indexing.
- **Simplified architecture:** Removing SQLite fallback eliminates complexity and ensures consistent behavior.

## Consequences

### Positive
- Shorter write stalls and fewer `database is locked` conditions during rebuilds and file-watcher updates.
- Better throughput for multi-repo and multi-threaded indexing workloads.
- Simplified codebase with no fallback logic.

### Negative
- Adds a native dependency (`rocksdict` wheels for Python; underlying RocksDB libraries).
- On-disk layout changes (`trigrams.rocksdb/` instead of `trigrams.db`); operators must reindex to populate the rocksdict store.
- Slightly more operational tuning surface (block cache, compaction options), though defaults are provided.

## Migration

- Install rocksdict extra: `pip install -e .[trigrams-rocksdict]` (or include in `server-full`); this pulls `rocksdict` wheels. System RocksDB libs should be present for best performance.
- Restart the server; it will use rocksdict exclusively.
- Run a trigram rebuild to populate the rocksdict store (Admin API or `sigil-rebuild-indexes`).
- Old `trigrams.db` files are no longer created or used.

## Alternatives Considered

- **Stay on SQLite** with more batching/backoff: reduced but did not eliminate lock contention under mixed workloads.
- **LMDB/LevelDB**: viable, but rocksdict offers better write throughput and tunable compaction for larger codebases.
- **Full-text engines (Elasticsearch/Solr)**: heavier operational footprint, not aligned with lightweight on-prem goals.

## References

- [ADR-002: Trigram-Based Text Indexing](adr-002-trigram-indexing.md) (superseded)
- rocksdict: https://github.com/huanxin/rocksdict
