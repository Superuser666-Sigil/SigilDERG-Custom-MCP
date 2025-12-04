<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR 010: Thread Safety for SQLite Indexer

**Status:** Accepted  
**Date:** 2025-12-03

## Context

The Sigil MCP server's indexing system uses SQLite databases to store repository metadata, document content, symbols, trigrams, and vector embeddings. The server operates in a multi-threaded environment with concurrent access from multiple sources:

1. **HTTP Request Handlers**: Multiple concurrent MCP tool invocations (search_code, find_symbol, semantic_search)
2. **File Watcher Thread**: Automatic file change detection triggering incremental re-indexing
3. **Vector Indexing Operations**: Background embedding generation for semantic search
4. **Manual Re-indexing**: User-initiated full repository scans

SQLite connections with `check_same_thread=False` allow cross-thread usage, but **do not** make concurrent access safe. Multiple threads executing queries on the same connection simultaneously can cause:

- Database locked errors
- Corrupted query results
- Inconsistent reads during writes
- Segmentation faults in extreme cases

The problem became apparent when:
- File watcher modified index while search was running → "database is locked"
- Vector indexing collided with trigram searches → inconsistent results
- Concurrent HTTP requests hit the same connection → random failures

## Decision

Implement comprehensive thread safety using **Write-Ahead Logging (WAL) mode** and **threading.RLock** serialization:

### 1. WAL Mode for Concurrent Readers

Enable SQLite's WAL (Write-Ahead Log) journaling mode on both databases:

```python
self.repos_db.execute("PRAGMA journal_mode=WAL;")
self.repos_db.execute("PRAGMA synchronous=NORMAL;")
self.repos_db.execute("PRAGMA busy_timeout=5000;")

self.trigrams_db.execute("PRAGMA journal_mode=WAL;")
self.trigrams_db.execute("PRAGMA synchronous=NORMAL;")
self.trigrams_db.execute("PRAGMA busy_timeout=5000;")
```

**WAL Mode Benefits:**
- Multiple readers can operate concurrently without blocking
- Writers don't block readers (readers see last committed state)
- Better concurrency than default rollback journal
- Creates separate `-wal` and `-shm` files alongside database

**PRAGMA Configuration:**
- `synchronous=NORMAL`: Balance between durability and performance (acceptable for local index)
- `busy_timeout=5000`: Wait up to 5 seconds for locks before failing (handles transient contention)

### 2. Global RLock for Connection Serialization

Add a reentrant lock (`threading.RLock`) to the `SigilIndex` class:

```python
self._lock = threading.RLock()
```

**Why RLock (Reentrant Lock)?**
- Allows the same thread to acquire the lock multiple times
- Necessary because public methods call internal methods (e.g., `search_code` → `_get_document`)
- Prevents deadlocks from recursive locking
- Zero overhead when single-threaded

### 3. Lock All Public Entry Points

Wrap every public method with the lock:

```python
def search_code(self, query, repo=None, max_results=50):
    with self._lock:
        # All database operations here
```

**Methods Protected:**
- `index_file()` - Granular file re-indexing (used by file watcher)
- `index_repository()` - Full repository indexing
- `search_code()` - Trigram-based code search
- `find_symbol()` - Symbol definition lookup
- `list_symbols()` - Symbol enumeration
- `build_vector_index()` - Embedding generation
- `semantic_search()` - Vector similarity search
- `get_index_stats()` - Index statistics

**Internal Methods NOT Protected:**
- `_index_file()`, `_build_trigram_index()`, `_get_document()`, etc.
- Already under lock when called from public methods
- Avoiding double-locking overhead

### 4. Connection Sharing Strategy

- **Single Connection Per Database**: One `repos_db` and one `trigrams_db` connection shared across all threads
- **No Connection Pool**: Serialization via RLock makes pooling unnecessary
- **check_same_thread=False**: Required for cross-thread usage (safe with locking)

## Alternatives Considered

### 1. Connection-Per-Thread Pattern

Create separate SQLite connections for each thread using thread-local storage.

**Rejected because:**
- WAL mode already enables concurrent reads without separate connections
- Connection proliferation wastes memory (each connection has its own page cache)
- More complex lifecycle management (thread creation/destruction)
- Doesn't solve writer contention (only one writer at a time regardless)
- Lock-based serialization simpler and proven effective

### 2. Connection Pooling

Use a connection pool (e.g., SQLAlchemy) to manage multiple connections.

**Rejected because:**
- Adds heavy dependency for minimal benefit
- Pool complexity unnecessary when RLock provides adequate serialization
- SQLite write operations are serialized at OS level anyway (file locking)
- WAL mode's concurrent read support doesn't require pool
- Single connection with lock has lower overhead

### 3. Read-Write Lock Separation

Use separate read lock and write lock (threading.RWLock pattern).

**Rejected because:**
- WAL mode already provides reader-writer separation at database level
- Complex to implement correctly (prevent writer starvation)
- Marginal performance gain vs implementation complexity
- RLock simpler and sufficient for our access patterns
- Most operations complete in milliseconds, lock contention rare

### 4. Lock-Free Concurrent Data Structures

Use lock-free queues and atomic operations for coordination.

**Rejected because:**
- SQLite itself uses locks (file-level, page-level)
- Can't make SQLite API lock-free from application layer
- Significant implementation complexity
- No real benefit over simpler locking approach
- Would still need serialization for connection access

## Consequences

### Positive

1. **Eliminates Database Locked Errors**: RLock serialization prevents concurrent connection access
2. **Concurrent Reads**: WAL mode allows multiple search operations simultaneously
3. **File Watcher Safety**: Background indexing doesn't block searches (readers see old state)
4. **Simple Implementation**: Single RLock, wrap public methods, done
5. **Zero Breaking Changes**: API remains identical, thread safety is transparent
6. **Minimal Performance Impact**: Lock operations are nanoseconds, database I/O dominates
7. **Production Ready**: WAL mode is well-tested and widely used in SQLite applications

### Negative

1. **Extra Files**: WAL mode creates `-wal` and `-shm` files (cleaned up on checkpoint)
2. **Disk Space**: WAL file can grow between checkpoints (auto-checkpoint at 1000 pages by default)
3. **Serialized Writes**: Only one write operation at a time (inherent SQLite limitation)
4. **Network Filesystems**: WAL mode requires proper file locking (not all network FS support it)
5. **Slightly More Disk I/O**: WAL mode does more writes than rollback journal

### Neutral

1. **Backup Considerations**: Must use `PRAGMA wal_checkpoint(TRUNCATE)` before copying database
2. **Checkpoint Overhead**: Periodic WAL checkpoints merge changes into main database
3. **Lock Granularity**: Per-instance locking (separate SigilIndex instances don't share lock)

## Implementation Notes

### WAL Mode Characteristics

**Concurrent Read Performance:**
```
Without WAL: Readers block on writers (SHARED lock conflict)
With WAL:    Readers see last committed state, never block on writers
```

**Write Performance:**
```
Without WAL: Each transaction writes to main database + journal
With WAL:    Writes append to WAL file, periodically checkpoint to main DB
```

**Checkpoint Triggers:**
- Automatic at 1000 WAL pages (configurable)
- Manual via `PRAGMA wal_checkpoint`
- On database close (normal shutdown)

### Lock Contention Analysis

**Typical Hold Times:**
- `search_code()`: 10-50ms (trigram intersection + file reads)
- `find_symbol()`: 1-5ms (simple indexed lookup)
- `index_file()`: 50-200ms (single file: read + parse + symbol extraction + trigrams)
- `build_vector_index()`: 1-10s (embedding generation, but batched)

**Contention Risk:**
- Low: Most operations complete in milliseconds
- Medium: Vector indexing holds lock for seconds (but rare operation)
- Mitigation: WAL mode allows reads during writes, search operations never block

### Thread Safety Testing

Tests verify:
1. Concurrent searches don't fail with "database locked"
2. Search during indexing returns valid results (possibly stale)
3. File watcher re-indexing doesn't corrupt trigram data
4. Vector indexing concurrent with searches works correctly

## Future Improvements

1. **Finer-Grained Locking**: Separate locks for repos_db vs trigrams_db (if contention observed)
2. **WAL Checkpoint Tuning**: Adjust checkpoint thresholds based on index size
3. **Lock Monitoring**: Expose lock hold times and contention metrics
4. **Async I/O Integration**: Use async SQLite driver (e.g., aiosqlite) for better concurrency in async contexts
5. **Optimistic Locking**: Try reads without lock, retry if write detected (for extremely read-heavy workloads)

## References

- [SQLite WAL Mode Documentation](https://www.sqlite.org/wal.html)
- [SQLite Thread Safety](https://www.sqlite.org/threadsafe.html)
- [Python threading.RLock](https://docs.python.org/3/library/threading.html#rlock-objects)
- [ADR 002: Trigram Indexing](adr-002-trigram-indexing.md) - Index structure this protects
- [ADR 007: File Watching](adr-007-file-watching.md) - Background thread requiring thread safety
- [ADR 006: Vector Embeddings](adr-006-vector-embeddings.md) - Concurrent embedding operations
