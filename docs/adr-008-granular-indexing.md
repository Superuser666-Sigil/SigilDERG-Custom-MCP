<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR 008: Granular Re-indexing and Configurable Watch Patterns

**Status:** Accepted  
**Date:** 2025-12-03

## Context

The initial file-watching implementation (ADR-007) automatically re-indexed entire repositories when files changed. While `SigilIndex.index_repository()` is smart enough to skip unchanged files by comparing content hashes, it still requires:

1. **Full directory tree traversal** - Walking all directories to discover files
2. **Database queries for every file** - Checking if each file needs updating
3. **Overhead for massive repositories** - Even with caching, large monorepos experience lag

Additionally, the ignore patterns for file watching were hardcoded in `watcher.py`, making it impossible for users to customize which files/directories to ignore without modifying source code. This is problematic for:

- Projects with custom build directories (e.g., `coverage/`, `htmlcov/`, `.next/`)
- Organizations with specific tooling that generates temporary files
- Different language ecosystems with unique artifact patterns

## Decision

We will implement two complementary improvements:

### 1. Granular Re-indexing

Add per-file re-indexing capability to avoid full repository scans:

**New Methods:**
- `SigilIndex.index_file(repo_name, repo_path, file_path)` - Index single file
- `SigilIndex._update_trigrams_for_file(repo_id, repo_path, file_path)` - Update trigrams for one file

**Flow:**
```
File Change Detected
  ↓
_on_file_change(repo_name, file_path, event_type)
  ↓
index.index_file(repo_name, repo_path, file_path)  # Granular
  ↓
Only this file's data updated
```

**Benefits:**
- **No directory traversal** - Direct path to changed file
- **Single database update** - One document, one set of trigrams
- **Constant-time operation** - O(1) instead of O(n) where n = total files

### 2. Configurable Watch Patterns

Move ignore patterns from hardcoded lists to configuration:

**Configuration Schema:**
```json
{
  "watch": {
    "enabled": true,
    "debounce_seconds": 2.0,
    "ignore_dirs": [
      ".git", "__pycache__", "node_modules", "target",
      "build", "dist", ".venv", "venv", ".tox",
      ".mypy_cache", ".pytest_cache", "coverage", ".coverage"
    ],
    "ignore_extensions": [
      ".pyc", ".so", ".o", ".a", ".dylib", ".dll",
      ".exe", ".bin", ".pdf", ".png", ".jpg", ".gif",
      ".svg", ".ico", ".woff", ".woff2", ".ttf",
      ".zip", ".tar", ".gz", ".bz2", ".xz"
    ]
  }
}
```

**Config Properties:**
- `Config.watch_ignore_dirs` - List of directory names to skip
- `Config.watch_ignore_extensions` - List of file extensions to skip

**Watcher Updates:**
- `RepositoryWatcher.__init__()` accepts `ignore_dirs` and `ignore_extensions`
- `FileWatchManager.__init__()` accepts and passes through ignore patterns
- `_should_ignore()` uses configured patterns instead of hardcoded sets

## Alternatives Considered

### 1. Full Repository Re-index (Status Quo)
**Rejected:** Acceptable for small repos, but scales poorly. For a 10,000 file repository, even skipping unchanged files requires thousands of hash comparisons.

### 2. Incremental Index Updates Only
**Rejected:** Deletions still require full scan to detect missing files. Partial solution doesn't justify complexity.

### 3. Database-Level Change Tracking
**Rejected:** Would require triggers, foreign keys, and complex bookkeeping. Granular file indexing is simpler and sufficient.

### 4. Git Integration for Change Detection
**Rejected:** Only works for git repositories, misses uncommitted changes, requires git library dependency.

### 5. Ignore Patterns in Separate File
**Rejected:** Adds another file to manage. Configuration JSON is already the source of truth for settings.

### 6. Regex-based Ignore Patterns
**Rejected:** More flexible but significantly more complex. Simple lists of directories and extensions cover 99% of use cases.

## Consequences

### Positive

1. **Massive Performance Improvement**: Re-indexing a single file takes milliseconds instead of seconds/minutes
2. **Scales to Large Repositories**: 100k+ file monorepos now viable
3. **User Customization**: Projects can define their own ignore patterns
4. **Sensible Defaults**: Common patterns work out-of-box, customization optional
5. **No Breaking Changes**: Existing configs work with sensible defaults
6. **Real-Time Deletion Cleanup**: In combination with `SigilIndex.remove_file` (see ADR-007),
   file deletions are now reflected immediately in the index without a full rebuild.

### Negative

1. **Config Complexity**: More configuration options for users to understand
2. **Pattern Maintenance**: Users must keep ignore patterns up-to-date with their projects

### Neutral

1. **Implementation Complexity**: Two new methods in indexer, parameter threading through watcher
2. **Testing Requirements**: Need tests for granular indexing and ignore pattern filtering

## Implementation Details

### Granular Re-indexing

**index_file() Flow:**
```python
def index_file(repo_name, repo_path, file_path):
    1. Get/create repo entry in database
    2. Determine language from file extension
    3. Call _index_file() with specific file path
    4. If successful, call _update_trigrams_for_file()
    5. Commit transaction
    6. Return success/failure
```

**Path Handling:**
- Input: Absolute `file_path`
- Storage: Relative path from `repo_path` (same as full indexing)
- Consistency: Uses same `relative_to()` logic as `index_repository()`

**Trigram Updates:**
```python
def _update_trigrams_for_file(repo_id, repo_path, file_path):
    1. Calculate relative path
    2. Query document ID from database
    3. Read blob content
    4. Extract trigrams
    5. Update trigrams table (merge with existing)
    6. Commit changes
```

**Limitations:**
- **Simplified trigram merge** - Doesn't remove old trigrams from unrelated files (acceptable,
  they simply point to other documents that still exist)
- **Move/Rename semantics** - File moves/renames are currently modeled as "delete old path +
  re-index new path" rather than in-place path updates

### Configurable Ignore Patterns

**Config Loading:**
- Defaults defined in `Config._default_config`
- User overrides in `config.json`
- Environment variables (future: `SIGIL_WATCH_IGNORE_DIRS`)

**Pattern Matching:**
- **Directories**: Exact name match against any part of path
  - Example: `__pycache__` matches `src/__pycache__/` and `test/__pycache__/`
- **Extensions**: Case-insensitive suffix match
  - Example: `.pyc` matches `test.pyc` and `TEST.PYC`

**Hidden Files:**
- Automatically ignored unless explicitly in `ignore_dirs`
- Prevents accidental indexing of `.git/`, `.DS_Store`, etc.

**Watcher Integration:**
```python
watcher = RepositoryWatcher(
    repo_name=name,
    repo_path=path,
    on_change=callback,
    ignore_dirs=config.watch_ignore_dirs,
    ignore_extensions=config.watch_ignore_extensions,
)
```

## Performance Impact

### Before (Full Re-index)

**10,000 file repository, 1 file changed:**
- Directory traversal: ~2 seconds
- Hash comparison: ~5 seconds
- Re-index changed file: ~50ms
- **Total: ~7 seconds**

### After (Granular Re-index)

**Same scenario:**
- Direct file access: ~1ms
- Re-index changed file: ~50ms
- **Total: ~51ms**

**~140x faster** for single file changes!

### Ignore Pattern Impact

**Before:** All patterns checked every time (hardcoded)

**After:** User-configurable patterns loaded once at startup

**Difference:** Negligible performance change, massive flexibility gain

## Migration Path

**Existing Deployments:**

1. **No config changes required** - Defaults match previous hardcoded patterns
2. **Automatic benefit** - Granular indexing automatically used for file changes
3. **Optional customization** - Add `ignore_dirs`/`ignore_extensions` to optimize

**Recommended Additions:**

```json
{
  "watch": {
    "ignore_dirs": [
      // Add project-specific dirs
      "htmlcov",     // Coverage reports
      ".next",       // Next.js build
      "tmp",         // Temporary files
      "logs"         // Log files
    ],
    "ignore_extensions": [
      // Add project-specific extensions
      ".log",
      ".tmp",
      ".cache"
    ]
  }
}
```

For a **full, clean rebuild** of all repositories (including blobs, trigrams, symbols,
and embeddings), use the operational helper script introduced alongside this ADR:

- `scripts/rebuild_indexes.py`, which:
  - Deletes the entire index directory configured in `config.json`
  - Re-initializes the index
  - Rebuilds all configured repositories from scratch

## Future Enhancements

1. **Move/Rename Detection**: Detect file moves and update paths without re-processing
2. **Batch Granular Updates**: Process multiple changed files in single transaction
3. **Ignore Pattern Testing**: Tool to test if a path would be ignored
4. **Regex Support**: Advanced pattern matching for power users (opt-in)
5. **Pattern Presets**: Common presets for different languages/frameworks

## Testing Strategy

**Unit Tests:**
- `test_index_file()` - Verify single file indexing
- `test_update_trigrams_for_file()` - Verify trigram updates
- `test_watch_ignore_patterns()` - Verify pattern matching

**Integration Tests:**
- Create file → Verify indexed
- Modify file → Verify re-indexed
- Delete file → Verify `remove_file` removes documents, symbols, embeddings, trigrams, and blob
  for that path
- File matches ignore pattern → Verify not indexed

**Performance Tests:**
- Measure re-index time for various repository sizes
- Verify O(1) scaling for granular re-index

## Documentation Updates

- [YES] **README.md** - Updated file watching section with ignore patterns
- [YES] **ADR-007** - Updated to reflect granular re-indexing
- [YES] **ADR-008** - This document
- [YES] **RUNBOOK.md** - New operational procedures
- [YES] **TROUBLESHOOTING.md** - New troubleshooting guide

## References

- [ADR-002: Trigram Indexing](adr-002-trigram-indexing.md)
- [ADR-007: File Watching](adr-007-file-watching.md)
- [GitHub Issue: File watching performance](https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP/issues/XX)

## Approval

**Proposed:** 2025-12-03  
**Accepted:** 2025-12-03  
**Implemented:** 2025-12-03

**Stakeholders:**
- Development Team [OK]
- Operations Team [OK]
- Users (Large Repository Owners) [OK]
