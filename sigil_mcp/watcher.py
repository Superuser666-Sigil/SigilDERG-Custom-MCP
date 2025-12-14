# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
File watching for automatic index updates.

Uses watchdog to monitor repository directories for changes and
automatically triggers re-indexing of modified files.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Callable, TYPE_CHECKING
from threading import Thread, Lock
from .ignore_utils import (
    load_gitignore,
    is_ignored_by_gitignore,
    load_include_patterns,
    should_ignore,
)
from .config import get_config

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None  # type: ignore
    FileSystemEventHandler = None  # type: ignore
    FileSystemEvent = None  # type: ignore
    if TYPE_CHECKING:
        from watchdog.observers import Observer  # type: ignore
        from watchdog.events import FileSystemEventHandler, FileSystemEvent  # type: ignore
    else:
        Observer = None  # type: ignore
        FileSystemEventHandler = object
        FileSystemEvent = object  # type: ignore

logger = logging.getLogger(__name__)

# Reduce noise from watchdog's internal inotify logging
# These DEBUG logs are very verbose and not useful for our purposes
# Set this early to prevent any watchdog DEBUG logs from appearing
for logger_name in [
    "watchdog.observers.inotify_buffer",
    "watchdog.observers.inotify",
    "watchdog.observers",
    "watchdog",
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


class RepositoryWatcher(FileSystemEventHandler):
    """Watches a repository directory for file changes."""
    
    def __init__(
        self,
        repo_name: str,
        repo_path: Path,
        on_change: Callable[[str, Path, str], None],
        debounce_seconds: float = 2.0,
        ignore_dirs: Optional[list[str]] = None,
        ignore_extensions: Optional[list[str]] = None,
        honor_gitignore: bool = True,
        repo_ignore_patterns: Optional[list[str]] = None,
    ):
        """
        Initialize repository watcher.
        
        Args:
            repo_name: Logical repository name
            repo_path: Path to repository root
            on_change: Callback function(repo_name, file_path, event_type)
            debounce_seconds: Delay before triggering re-index (batches changes)
        """
        super().__init__()
        self.repo_name = repo_name
        self.repo_path = repo_path
        self.on_change = on_change
        self.debounce_seconds = debounce_seconds
        self.ignore_dirs = set(ignore_dirs or [])
        self.ignore_extensions = set(ignore_extensions or [])
        # Optionally load per-repo .gitignore/include patterns. honor_gitignore
        # allows disabling repo-level ignores (useful for per-repo overrides).
        self._honor_gitignore = bool(honor_gitignore)
        if self._honor_gitignore:
            try:
                self._gitignore_patterns = load_gitignore(repo_path)
            except Exception:
                self._gitignore_patterns = []
            try:
                self._include_patterns = load_include_patterns(repo_path)
            except Exception:
                self._include_patterns = []
        else:
            self._gitignore_patterns = []
            self._include_patterns = []
        # Optional per-repo ignore patterns provided via admin API / config
        self._repo_ignore_patterns = list(repo_ignore_patterns or [])
        
        # Track pending changes to batch updates
        self.pending_changes: Dict[str, tuple[Path, str, float]] = {}
        self.lock = Lock()
        self.processing_thread: Optional[Thread] = None
        self.running = True
        
        # Start background thread to process changes
        self._start_processing_thread()
    
    def _start_processing_thread(self):
        """Start background thread to process batched changes."""
        self.processing_thread = Thread(target=self._process_changes, daemon=True)
        self.processing_thread.start()
    
    def _process_changes(self):
        """Background thread that processes batched file changes."""
        while self.running:
            time.sleep(0.5)  # Check every 500ms
            
            with self.lock:
                now = time.time()
                ready_changes = [
                    (path, event_type)
                    for path, (path_obj, event_type, timestamp) in self.pending_changes.items()
                    if now - timestamp >= self.debounce_seconds
                ]
                
                # Remove processed changes
                for path, _ in ready_changes:
                    del self.pending_changes[path]
            
            # Process ready changes (outside lock to avoid blocking)
            for path_str, event_type in ready_changes:
                try:
                    path_obj = Path(path_str)
                    self.on_change(self.repo_name, path_obj, event_type)
                except Exception as e:
                    logger.error(
                        f"Error processing change for {path_str} in {self.repo_name}: {e}"
                    )
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if file should be ignored, honoring configured ignore rules."""
        try:
            cfg = get_config()
        except Exception:
            cfg = None

        return should_ignore(
            path,
            self.repo_path,
            config_ignore_patterns=(cfg.index_ignore_patterns if cfg is not None else None),
            repo_ignore_patterns=getattr(self, '_repo_ignore_patterns', None),
            include_patterns=getattr(self, '_include_patterns', None),
            gitignore_patterns=getattr(self, '_gitignore_patterns', None),
            ignore_dirs=self.ignore_dirs,
            ignore_extensions=self.ignore_extensions,
        )
    
    def _schedule_change(self, path_str: str, event_type: str):
        """Schedule a file change for processing (with debouncing)."""
        try:
            path = Path(path_str).resolve()
            
            # Ensure path is under repo
            try:
                path.relative_to(self.repo_path)
            except ValueError:
                return  # Outside repo, ignore
            
            # For deletions the file will not exist anymore, but we still
            # need to schedule the event so callers can react to deletes.
            should_schedule = False
            if event_type == "deleted":
                should_schedule = True
                skip_ignore_check = True
            elif path.is_file():
                should_schedule = True
                skip_ignore_check = False

            # For deletions, the file may not exist so avoid stat-based ignore checks
            if should_schedule and (skip_ignore_check or not self._should_ignore(path)):
                with self.lock:
                    # Update or add pending change
                    self.pending_changes[str(path)] = (path, event_type, time.time())
        except Exception as e:
            # Avoid noisy debug logging in hot watcher path; surface only as info
            logger.info(f"Error scheduling change for {path_str}: {e}")
    
    def _should_ignore_path(self, path_str: str) -> bool:
        """Quick check to ignore paths before any processing."""
        normalized = path_str.replace('\\', '/')
        parts = normalized.split('/')
        # Fast include check: if explicitly included, do not ignore
        try:
            p = Path(normalized)
            if getattr(self, '_include_patterns', None) and is_ignored_by_gitignore(p, self.repo_path, getattr(self, '_include_patterns')):
                return False
            if self._gitignore_patterns and is_ignored_by_gitignore(p, self.repo_path, self._gitignore_patterns):
                return True
        except Exception:
            pass
        
        # Check against configured ignore directories
        # This prevents watchdog from even processing events for these dirs
        if self.ignore_dirs:
            # Normalize ignore dirs (handle with/without leading dot)
            normalized_ignore = set()
            for ignore_dir in self.ignore_dirs:
                normalized_ignore.add(ignore_dir)
                if ignore_dir.startswith('.'):
                    normalized_ignore.add(ignore_dir.lstrip('.'))
                else:
                    normalized_ignore.add(f'.{ignore_dir}')
            
            # Check if any path component matches an ignored directory
            if any(part in normalized_ignore or f'.{part}' in normalized_ignore 
                   for part in parts):
                return True
        
        return False
    
    def on_modified(self, event: FileSystemEvent):  # type: ignore
        """Called when a file is modified."""
        # Ignore .git directory entirely - don't even process these events
        if self._should_ignore_path(str(event.src_path)):
            return
        if not event.is_directory:
            self._schedule_change(str(event.src_path), "modified")
    
    def on_created(self, event: FileSystemEvent):  # type: ignore
        """Called when a file is created."""
        # Ignore .git directory entirely - don't even process these events
        if self._should_ignore_path(str(event.src_path)):
            return
        if not event.is_directory:
            self._schedule_change(str(event.src_path), "created")
    
    def on_deleted(self, event: FileSystemEvent):  # type: ignore
        """Called when a file is deleted."""
        # Ignore .git directory entirely - don't even process these events
        if self._should_ignore_path(str(event.src_path)):
            return
        if not event.is_directory:
            self._schedule_change(str(event.src_path), "deleted")
    
    def stop(self):
        """Stop the processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)


class FileWatchManager:
    """Manages file watchers for multiple repositories."""
    
    def __init__(
        self,
        on_change: Callable[[str, Path, str], None],
        ignore_dirs: Optional[list[str]] = None,
        ignore_extensions: Optional[list[str]] = None,
    ):
        """
        Initialize file watch manager.
        
        Args:
            on_change: Callback function(repo_name, file_path, event_type)
            ignore_dirs: Directories to ignore when watching
            ignore_extensions: File extensions to ignore when watching
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning(
                "watchdog not available - file watching disabled. "
                "Install with: pip install sigil-mcp-server[watch]"
            )
        
        self.on_change = on_change
        self.ignore_dirs = ignore_dirs or []
        self.ignore_extensions = ignore_extensions or []
        self.observer: Optional[Observer] = None  # type: ignore
        self.watchers: Dict[str, RepositoryWatcher] = {}
        self.enabled = WATCHDOG_AVAILABLE
    
    def start(self):
        """Start the file watch manager."""
        if not self.enabled or Observer is None:
            return
        
        self.observer = Observer()
        if self.observer is not None:
            self.observer.start()
            logger.info("File watching enabled")
    
    def watch_repository(
        self,
        repo_name: str,
        repo_path: Path,
        recursive: bool = True,
        honor_gitignore: bool = True,
        repo_ignore_patterns: Optional[list[str]] = None,
    ):
        """
        Start watching a repository for changes.
        
        Args:
            repo_name: Logical repository name
            repo_path: Path to repository root
            recursive: Watch subdirectories recursively
        """
        if not self.enabled or not self.observer:
            return
        
        if repo_name in self.watchers:
            logger.debug(f"Already watching {repo_name}")
            return
        
        try:
            watcher = RepositoryWatcher(
                repo_name=repo_name,
                repo_path=repo_path,
                on_change=self.on_change,
                ignore_dirs=self.ignore_dirs,
                ignore_extensions=self.ignore_extensions,
                honor_gitignore=honor_gitignore,
                repo_ignore_patterns=repo_ignore_patterns,
            )
            # Ensure server.REPOS knows about this repository so callbacks
            # that rely on server.REPOS (e.g., _on_file_change) can resolve
            # the repo root when running inside tests or external watchers.
            try:
                # local import to avoid circular top-level import
                from sigil_mcp import server as _server
                if repo_name not in getattr(_server, "REPOS", {}):
                    _server.REPOS[repo_name] = repo_path
            except Exception:
                # Non-fatal: if we can't update server.REPOS, continue watching
                pass
            
            self.observer.schedule(watcher, str(repo_path), recursive=recursive)
            self.watchers[repo_name] = watcher
            
            logger.info(f"Watching {repo_name} at {repo_path}")
        except Exception as e:
            logger.error(f"Failed to watch {repo_name}: {e}")
    
    def unwatch_repository(self, repo_name: str):
        """Stop watching a repository."""
        if repo_name in self.watchers:
            watcher = self.watchers[repo_name]
            watcher.stop()
            del self.watchers[repo_name]
            logger.info(f"Stopped watching {repo_name}")
    
    def stop(self):
        """Stop all file watchers."""
        if not self.enabled or not self.observer:
            return
        
        # Stop all watchers
        for watcher in self.watchers.values():
            watcher.stop()
        
        self.watchers.clear()
        
        # Stop observer
        self.observer.stop()
        self.observer.join(timeout=5.0)
        
        logger.info("File watching stopped")
    
    def is_watching(self, repo_name: str) -> bool:
        """Check if a repository is being watched."""
        return repo_name in self.watchers
