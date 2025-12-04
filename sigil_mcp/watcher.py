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
        # Always skip hidden files/dirs (top-level safety)
        if any(part.startswith('.') for part in path.parts):
            return True
        
        # Check directories from config
        if self.ignore_dirs:
            # Normalize: treat ".git" and "git" sensibly
            normalized_dirs = set()
            for name in self.ignore_dirs:
                normalized_dirs.add(name)
                # also support matching without leading dot
                if name.startswith('.'):
                    normalized_dirs.add(name.lstrip('.'))
            
            if any(
                part in normalized_dirs or f".{part}" in normalized_dirs
                for part in path.parts
            ):
                return True
        
        # Check extensions from config
        if self.ignore_extensions:
            ext = path.suffix.lower()
            if ext in self.ignore_extensions:
                return True
        
        return False
    
    def _schedule_change(self, path_str: str, event_type: str):
        """Schedule a file change for processing (with debouncing)."""
        try:
            path = Path(path_str).resolve()
            
            # Ensure path is under repo
            try:
                path.relative_to(self.repo_path)
            except ValueError:
                return  # Outside repo, ignore
            
            if path.is_file() and not self._should_ignore(path):
                with self.lock:
                    # Update or add pending change
                    self.pending_changes[str(path)] = (path, event_type, time.time())
                    logger.debug(
                        f"Scheduled {event_type} for {path.relative_to(self.repo_path)}"
                    )
        except Exception as e:
            logger.debug(f"Error scheduling change for {path_str}: {e}")
    
    def on_modified(self, event: FileSystemEvent):  # type: ignore
        """Called when a file is modified."""
        if not event.is_directory:
            self._schedule_change(str(event.src_path), "modified")
    
    def on_created(self, event: FileSystemEvent):  # type: ignore
        """Called when a file is created."""
        if not event.is_directory:
            self._schedule_change(str(event.src_path), "created")
    
    def on_deleted(self, event: FileSystemEvent):  # type: ignore
        """Called when a file is deleted."""
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
        recursive: bool = True
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
            )
            
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
