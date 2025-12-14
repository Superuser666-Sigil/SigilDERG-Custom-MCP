<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR 010: Thread Safety for SQLite Indexer

**Status:** Superseded by [ADR-017: rocksdict Trigram Store](adr-017-rocksdb-trigram-store.md); SQLite-backed trigram/indexer paths are no longer used in the current codebase.  
**Date:** 2025-12-03

## Context

The Sigil MCP server's indexing system uses SQLite databases to store repository metadata and symbols. The trigram index now uses rocksdict exclusively (see ADR-017). The server operates in a multi-threaded environment with concurrent access from multiple sources:

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

Implement comprehensive thread safety for the remaining SQLite metadata database using **Write-Ahead Logging (WAL) mode** and **threading.RLock** serialization. (Trigram postings have moved to rocksdict; this ADR is retained for historical context on the SQLite metadata DB.)

### 1. WAL Mode for Concurrent Readers

Enable SQLite's WAL (Write-Ahead Log) journaling mode on the repos database:

```python
self.repos_db.execute("PRAGMA journal_mode=WAL;")
self.repos_db.execute("PRAGMA synchronous=NORMAL;")
self.repos_db.execute("PRAGMA busy_timeout=5000;")
```

**WAL Mode Benefits (at SQLite level):**
- Multiple readers can operate concurrently without blocking (if they could access connection)
- Writers don't block readers (readers see last committed state)
- Better concurrency than default rollback journal
- Creates separate `-wal` and `-shm` files alongside database

**PRAGMA Configuration:**
- `synchronous=NORMAL`: Balance between durability and performance (acceptable for local index)
- `busy_timeout=5000`: Wait up to 5 seconds for locks before failing (handles transient contention)

**Important Reality Check:**
WAL mode provides concurrent read capability *at the database level*, but our RLock serialization (below) means all operations are serialized *at the Python layer*. This means in practice, reads will block on long writes even though SQLite could handle them concurrently. This is a deliberate simplicity tradeoff - see "Concurrency Reality" section below.

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

- **Single Connection Per Database**: One `repos_db` connection shared across all threads
- **No Connection Pool**: Serialization via RLock makes pooling unnecessary
- **check_same_thread=False**: Required for cross-thread usage (safe with locking)

### 5. Concurrency Reality

**What WAL Mode Provides:**
SQLite with WAL mode can handle multiple concurrent readers and one writer simultaneously. A long write operation won't block read queries.

**What RLock Serialization Does:**
Our implementation wraps all public methods with a single `RLock`, which serializes all database access at the Python layer *before* it reaches SQLite.

**Practical Impact:**
```python
# Thread 1: Long write operation
with self._lock:
    index.index_repository(...)  # Takes 10 seconds

# Thread 2: Quick read query (waits for Thread 1)
with self._lock:  # Blocks here until Thread 1 releases lock
    index.search_code(...)  # Can't proceed even though WAL could handle it
```

**Why This Tradeoff?**
- **Simplicity**: Single lock is trivial to reason about, no deadlock risk
- **Correctness**: Impossible to have race conditions with full serialization
- **Performance**: Most operations complete in milliseconds; lock contention is rare
- **Current Scale**: Adequate for typical usage (few concurrent requests)
- **SQLite Limitation**: Only one writer at a time anyway (serialization unavoidable)

**When You'd Need True Concurrent Reads:**
- High-traffic server with many simultaneous search requests
- Long-running write operations blocking latency-sensitive reads
- Multiple readers querying different databases simultaneously

**How to Achieve True Concurrency (Future):**
- Connection-per-thread pattern (thread-local storage)
- Read-write lock separation (shared lock for reads, exclusive for writes)
- Separate locks for `repos_db` vs `trigrams_db`

## Alternatives Considered

### 1. Connection-Per-Thread Pattern

Create separate SQLite connections for each thread using thread-local storage.

**Rejected because:**
- Current approach already serializes all access; connection-per-thread wouldn't improve concurrency without also changing locking strategy
- Connection proliferation wastes memory (each connection has its own page cache)
- More complex lifecycle management (thread creation/destruction)
- Doesn't solve writer contention (only one writer at a time regardless)
- Would need read-write lock separation to actually benefit from multiple connections
- Lock-based serialization simpler and proven effective for current scale

### 2. Connection Pooling

Use a connection pool (e.g., SQLAlchemy) to manage multiple connections.

**Rejected because:**
- Adds heavy dependency for minimal benefit
- Pool complexity unnecessary when RLock already provides full serialization
- SQLite write operations are serialized at OS level anyway (file locking)
- Wouldn't improve concurrency without changing locking strategy (same RLock bottleneck)
- Single connection with lock has lower overhead
- WAL mode's concurrent read support doesn't require pool
- Single connection with lock has lower overhead

### 3. Read-Write Lock Separation

Use separate read lock and write lock (threading.RWLock pattern) to allow concurrent reads.

**Rejected for v1 because:**
- Adds significant implementation complexity (multiple lock types, careful ordering)
- Risk of writer starvation or reader starvation if not balanced correctly
- Debugging difficulty (deadlock potential with complex lock interactions)
- Current serialized approach is adequate for typical load
- Most operations complete in milliseconds; lock contention is rare
- **May revisit**: If profiling shows read lock contention is a bottleneck, this would be the next evolution

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
2. **Simple Mental Model**: All operations are serialized - easy to reason about correctness
3. **File Watcher Safety**: Background indexing doesn't corrupt data or cause conflicts
4. **Simple Implementation**: Single RLock, wrap public methods, done
5. **Zero Breaking Changes**: API remains identical, thread safety is transparent
6. **Minimal Performance Impact**: Lock operations are nanoseconds, database I/O dominates
7. **Production Ready**: Proven pattern, no race conditions possible
8. **WAL Mode Foundation**: Enables future concurrent read optimization without changing SQLite setup

### Negative

1. **Serialized Operations**: All operations queue behind a single lock (no concurrent reads in practice)
2. **Long Write Blocking**: Repository indexing (10s) blocks quick searches (10ms)
3. **Extra Files**: WAL mode creates `-wal` and `-shm` files (cleaned up on checkpoint)
4. **Disk Space**: WAL file can grow between checkpoints (auto-checkpoint at 1000 pages by default)
5. **Network Filesystems**: WAL mode requires proper file locking (not all network FS support it)
6. **Underutilized WAL**: WAL's concurrent read capability not exploited with current locking

### Neutral

1. **Backup Considerations**: Must use `PRAGMA wal_checkpoint(TRUNCATE)` before copying database
2. **Checkpoint Overhead**: Periodic WAL checkpoints merge changes into main database
3. **Lock Granularity**: Per-instance locking (separate SigilIndex instances don't share lock)

## Implementation Notes

### WAL Mode Characteristics

**Concurrent Read Performance (Potential):**
```
Without WAL: Readers block on writers (SHARED lock conflict)
With WAL:    Readers *could* see last committed state without blocking
             (but current RLock serializes access before reaching SQLite)
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

**Typical Hold Times (with RLock serialization):**
- `search_code()`: 10-50ms (trigram intersection + file reads)
- `find_symbol()`: 1-5ms (simple indexed lookup)
- `index_file()`: 50-200ms (single file: read + parse + symbol extraction + trigrams)
- `build_vector_index()`: 1-10s (embedding generation, but batched)

**Actual Contention Behavior:**
- **Low Load**: Single operations complete quickly, no contention
- **Concurrent Searches**: Queue behind RLock, execute sequentially (each 10-50ms)
- **Search During Indexing**: Searches wait for full indexing operation to complete
- **Practical Impact**: Acceptable for typical usage (few concurrent users, infrequent indexing)

**Why This Is Adequate:**
- Most workloads are read-heavy with infrequent writes
- Individual operations are fast (milliseconds)
- Background indexing happens infrequently
- Correctness and simplicity outweigh marginal throughput gains

### Thread Safety Testing

Tests verify:
1. Concurrent searches don't fail with "database locked"
2. Search during indexing returns valid results (possibly stale)
3. File watcher re-indexing doesn't corrupt trigram data
4. Vector indexing concurrent with searches works correctly

## Future Improvements

1. **Read-Write Lock Pattern**: Implement shared/exclusive locking to enable true concurrent reads
   - Shared lock for read operations (multiple readers allowed)
   - Exclusive lock for write operations (blocks all others)
   - Would fully utilize WAL mode's concurrent read capability
   - Requires careful implementation to avoid deadlocks and starvation

2. **Connection-Per-Thread**: Thread-local connections for parallel read access
   - Each thread gets its own connection
   - Coordinate writes through separate queue or exclusive lock
   - Better utilization of multi-core systems
   - Higher memory usage (page cache per connection)

3. **Separate Database Locks**: Split `repos_db` and `trigrams_db` locks
   - Allow symbol search while trigram search runs
   - Reduces contention when operations touch different databases
   - Simple incremental improvement

4. **Lock Monitoring**: Expose lock hold times and contention metrics
   - Track wait times, hold times, contention frequency
   - Help identify performance bottlenecks
   - Guide future optimization decisions

5. **Async I/O Integration**: Use async SQLite driver (e.g., aiosqlite)
   - Better concurrency in async contexts
   - Non-blocking I/O for embedding operations
   - Requires async/await refactoring

6. **WAL Checkpoint Tuning**: Adjust checkpoint thresholds based on index size
   - Larger indexes may benefit from less frequent checkpoints
   - Balance between WAL size and checkpoint overhead

## References

- [SQLite WAL Mode Documentation](https://www.sqlite.org/wal.html)
- [SQLite Thread Safety](https://www.sqlite.org/threadsafe.html)
- [Python threading.RLock](https://docs.python.org/3/library/threading.html#rlock-objects)
- [ADR 002: Trigram Indexing](adr-002-trigram-indexing.md) - Index structure this protects
- [ADR 007: File Watching](adr-007-file-watching.md) - Background thread requiring thread safety
- [ADR 006: Vector Embeddings](adr-006-vector-embeddings.md) - Concurrent embedding operations
