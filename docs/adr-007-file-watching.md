<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR 007: File Watching for Automatic Index Updates

**Status:** Accepted  
**Date:** 2025-12-03

## Context

The Sigil MCP server maintains an index of repository files to enable fast code search and semantic features. Initially, this index is built manually and becomes stale as files change. Users must remember to manually re-index repositories after making changes, which is inconvenient for development workflows.

A file-watching mechanism would automatically detect file changes and trigger incremental index updates, keeping the index current without manual intervention.

## Decision

We will implement file watching using the **watchdog** library with the following characteristics:

### Architecture

1. **Optional Dependency**: Watchdog is an optional dependency (`pip install sigil-mcp-server[watch]`) to keep the base installation minimal
2. **Debouncing**: Changes are batched with a configurable delay (default 2 seconds) to avoid excessive re-indexing during bulk operations
3. **Background Processing**: A dedicated thread processes file changes asynchronously to avoid blocking the main server
4. **Granular Re-indexing**: Individual files are re-indexed when modified/created, avoiding full repository scans
5. **Configurable Filtering**: Ignore patterns for directories and file extensions can be customized per deployment

### Features

- **Automatic Detection**: Monitors file creation, modification, and deletion
- **Granular Updates**: Only the changed file is re-indexed, not the entire repository
- **Smart Filtering**: Configurable ignore patterns for directories and file extensions
- **Graceful Degradation**: Server works normally without watchdog installed (file watching simply disabled)
- **Per-Repository Watching**: Each configured repository gets its own watcher
- **Clean Shutdown**: Watchers are properly stopped when server shuts down

### Configuration

```json
{
  "watch": {
    "enabled": true,
    "debounce_seconds": 2.0,
    "ignore_dirs": [".git", "__pycache__", "node_modules", "build", "coverage"],
    "ignore_extensions": [".pyc", ".so", ".pdf", ".png", ".jpg"]
  }
}
```

Environment variables:
- `SIGIL_MCP_WATCH_ENABLED=true|false`
- `SIGIL_MCP_WATCH_DEBOUNCE=2.0`

## Alternatives Considered

### 1. Polling-Based Monitoring
**Rejected**: Higher resource usage, slower detection, and less reliable than event-based watching.

### 2. Built-in File Watching (inotify/FSEvents)
**Rejected**: Platform-specific, requires more code, and watchdog already provides cross-platform abstraction.

### 3. Git Hook Integration
**Rejected**: Only detects changes committed through git, misses file system operations and non-git workflows.

### 4. Mandatory Watchdog Dependency
**Rejected**: Adds unnecessary bloat for deployments that don't need file watching (e.g., static documentation, CI/CD).

## Consequences

### Positive

1. **Better Developer Experience**: Index stays current automatically during development
2. **No Manual Re-indexing**: Users don't need to remember to rebuild indexes
3. **Real-time Updates**: Changes are detected within seconds (after debounce)
4. **Optional Feature**: Zero cost for users who don't need it
5. **Efficient Batching**: Debouncing prevents excessive re-indexing during bulk operations

### Negative

1. **Additional Dependency**: Requires watchdog for full functionality
2. **Resource Usage**: Background threads and file system monitoring consume some resources
3. **Complexity**: More moving parts to test and maintain
4. **Platform Variations**: watchdog behavior may vary slightly across OS platforms

### Neutral

1. **Configuration Overhead**: Users need to understand watch settings
2. **Testing Complexity**: File watching requires time-based integration tests

## Implementation Notes

### File Change Flow

```
File Change on Disk
  ↓
watchdog Observer detects event
  ↓
RepositoryWatcher.on_modified/created/deleted
  ↓
_schedule_change (with debouncing)
  ↓
Background thread waits for debounce period
  ↓
on_change callback triggered
  ↓
_on_file_change(repo_name, file_path, event_type)
  ↓
┌──────────────────────────────┬─────────────────────────────────────────────┐
│ event_type in {"created",    │ event_type == "deleted"                    │
│ "modified"}                  │                                           │
├──────────────────────────────┼─────────────────────────────────────────────┤
│ index.index_file(...)        │ index.remove_file(...)                     │
│ - Re-index single file       │ - Remove documents, symbols, embeddings,   │
│ - Update trigrams/symbols    │   trigrams, and blob for that file        │
└──────────────────────────────┴─────────────────────────────────────────────┘
  ↓
Index updated incrementally and kept consistent with on-disk state
```

### Ignored Patterns

The watcher uses configurable ignore patterns (defaults shown):
- **Directories**: `.git`, `__pycache__`, `node_modules`, `target`, `build`, `dist`, `.venv`, `venv`, `.tox`, `.mypy_cache`, `.pytest_cache`, `coverage`, `.coverage`
- **Extensions**: `.pyc`, `.so`, `.o`, `.a`, `.dylib`, `.dll`, `.exe`, `.bin`, `.pdf`, `.png`, `.jpg`, `.gif`, `.svg`, `.ico`, `.woff`, `.woff2`, `.ttf`, `.zip`, `.tar`, `.gz`, `.bz2`, `.xz`
- **Hidden files**: Files/directories starting with `.` (except those explicitly in ignore_dirs)

These can be customized in `config.json` to match project-specific needs.

### Performance Considerations

- **Granular Re-indexing**: Only the changed file is processed, avoiding directory tree scans
- **Debouncing**: 2-second default prevents excessive re-indexing during saves
- **Recursive Watching**: Full directory trees are monitored efficiently by the OS
- **Batch Processing**: Multiple changes in quick succession are batched together
- **Background Execution**: Re-indexing and deletion cleanup don't block MCP requests
- **Deletion Handling**: File deletions trigger `SigilIndex.remove_file`, which removes all
  index data for that file (documents, symbols, embeddings, trigrams, and blob content)

## Future Improvements

1. **Smart Re-indexing for Moves/Renames**: Detect file moves/renames and update paths without
   re-processing content (build on top of existing granular indexing and deletion)
2. **Vector Index Updates**: Efficiently update embeddings for only changed chunks
3. **Watch Statistics**: Expose metrics about file changes and re-indexing operations via MCP tool
4. **Rate Limiting**: Protect against excessive events from mass file operations

## References

- [watchdog documentation](https://python-watchdog.readthedocs.io/)
- [ADR 002: Trigram Indexing](adr-002-trigram-indexing.md)
- [ADR 006: Vector Embeddings](adr-006-vector-embeddings.md)
- `scripts/rebuild_indexes.py` helper script for full index wipe-and-rebuild in operational workflows
