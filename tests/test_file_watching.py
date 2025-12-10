# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Integration tests for file watching functionality.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch


class TestFileWatching:
    """Test file watching and automatic re-indexing."""
    
    @pytest.fixture
    def watchdog_available(self):
        """Ensure watchdog is available for tests."""
        try:
            import watchdog  # noqa: F401
            return True
        except ImportError:
            pytest.skip("watchdog not installed")
    
    def test_file_watch_manager_initialization(self, watchdog_available):
        """Test FileWatchManager initialization."""
        from sigil_mcp.watcher import FileWatchManager
        
        on_change = Mock()
        manager = FileWatchManager(
            on_change=on_change,
            ignore_dirs=[".git", "__pycache__"],
            ignore_extensions=[".pyc", ".so"]
        )
        
        assert manager.enabled is True
        assert manager.on_change is on_change
        assert ".git" in manager.ignore_dirs
        assert ".pyc" in manager.ignore_extensions
    
    def test_file_watch_manager_without_watchdog(self):
        """Test FileWatchManager gracefully handles missing watchdog."""
        with patch('sigil_mcp.watcher.WATCHDOG_AVAILABLE', False):
            from sigil_mcp.watcher import FileWatchManager
            
            on_change = Mock()
            manager = FileWatchManager(on_change=on_change)
            
            assert manager.enabled is False
            
            # Should not raise errors
            manager.start()
            manager.watch_repository("test", Path("/fake/path"))
            manager.stop()
    
    def test_watch_repository(self, watchdog_available, test_repo_path):
        """Test watching a repository."""
        from sigil_mcp.watcher import FileWatchManager
        
        on_change = Mock()
        manager = FileWatchManager(on_change=on_change)
        manager.start()
        
        try:
            manager.watch_repository("test_repo", test_repo_path)
            assert manager.is_watching("test_repo")
        finally:
            manager.stop()
    
    def test_unwatch_repository(self, watchdog_available, test_repo_path):
        """Test unwatching a repository."""
        from sigil_mcp.watcher import FileWatchManager
        
        on_change = Mock()
        manager = FileWatchManager(on_change=on_change)
        manager.start()
        
        try:
            manager.watch_repository("test_repo", test_repo_path)
            assert manager.is_watching("test_repo")
            
            manager.unwatch_repository("test_repo")
            assert not manager.is_watching("test_repo")
        finally:
            manager.stop()
    
    def test_file_modification_triggers_callback(
        self, watchdog_available, test_repo_path
    ):
        """Test that file modifications trigger the callback."""
        from sigil_mcp.watcher import FileWatchManager
        
        on_change = Mock()
        manager = FileWatchManager(
            on_change=on_change,
            ignore_dirs=[],
            ignore_extensions=[]
        )
        manager.start()
        
        try:
            manager.watch_repository("test_repo", test_repo_path)
            
            # Modify a file
            test_file = test_repo_path / "main.py"
            original_content = test_file.read_text()
            test_file.write_text(original_content + "\n# Modified")
            
            # Wait for debounce + processing
            time.sleep(3.0)
            
            # Check callback was called
            assert on_change.called
            call_args = on_change.call_args
            assert call_args[0][0] == "test_repo"  # repo_name
            assert call_args[0][1] == test_file  # file_path
            assert call_args[0][2] in ["modified", "created"]  # event_type
        finally:
            manager.stop()
    
    def test_file_creation_triggers_callback(
        self, watchdog_available, test_repo_path
    ):
        """Test that file creation triggers the callback."""
        from sigil_mcp.watcher import FileWatchManager
        
        on_change = Mock()
        manager = FileWatchManager(on_change=on_change)
        manager.start()
        
        try:
            manager.watch_repository("test_repo", test_repo_path)
            
            # Create a new file
            new_file = test_repo_path / "new_module.py"
            new_file.write_text("def new_function():\n    pass\n")
            
            # Wait for debounce + processing
            time.sleep(3.0)
            
            # Check callback was called
            assert on_change.called
            call_args = on_change.call_args
            assert call_args[0][0] == "test_repo"
            assert call_args[0][1] == new_file
            assert call_args[0][2] in {"created", "modified"}
        finally:
            manager.stop()
            if new_file.exists():
                new_file.unlink()
    
    def test_file_deletion_triggers_callback(
        self, watchdog_available, test_repo_path
    ):
        """Test that file deletion triggers the callback."""
        from sigil_mcp.watcher import FileWatchManager
        
        # Create a temporary file
        temp_file = test_repo_path / "temp.py"
        temp_file.write_text("# Temporary file")
        
        on_change = Mock()
        manager = FileWatchManager(on_change=on_change)
        manager.start()
        
        try:
            manager.watch_repository("test_repo", test_repo_path)
            
            # Give watcher time to initialize
            time.sleep(0.5)
            
            # Delete the file
            temp_file.unlink()
            
            # Wait (with polling) for debounce + processing
            timeout = time.time() + 5.0
            while not on_change.called and time.time() < timeout:
                time.sleep(0.1)
            
            # Check callback was called at least once
            assert on_change.called
            call_args = on_change.call_args
            assert call_args[0][0] == "test_repo"
            assert call_args[0][2] == "deleted"
        finally:
            manager.stop()
    
    def test_ignored_files_not_watched(
        self, watchdog_available, test_repo_path
    ):
        """Test that ignored files don't trigger callbacks."""
        from sigil_mcp.watcher import FileWatchManager
        
        on_change = Mock()
        manager = FileWatchManager(
            on_change=on_change,
            ignore_extensions=[".pyc", ".tmp"]
        )
        manager.start()
        
        try:
            manager.watch_repository("test_repo", test_repo_path)
            
            # Create an ignored file
            ignored_file = test_repo_path / "test.pyc"
            ignored_file.write_bytes(b"fake bytecode")
            
            # Wait for potential callback
            time.sleep(3.0)
            
            # Callback should not have been called
            assert not on_change.called
        finally:
            manager.stop()
            if ignored_file.exists():
                ignored_file.unlink()
    
    def test_ignored_directories_not_watched(
        self, watchdog_available, test_repo_path
    ):
        """Test that files in ignored directories don't trigger callbacks."""
        from sigil_mcp.watcher import FileWatchManager
        
        on_change = Mock()
        manager = FileWatchManager(
            on_change=on_change,
            ignore_dirs=["__pycache__", ".git"]
        )
        manager.start()
        
        try:
            manager.watch_repository("test_repo", test_repo_path)
            
            # Create ignored directory and file
            ignored_dir = test_repo_path / "__pycache__"
            ignored_dir.mkdir(exist_ok=True)
            ignored_file = ignored_dir / "test.pyc"
            ignored_file.write_bytes(b"fake bytecode")
            
            # Wait for potential callback
            time.sleep(3.0)
            
            # Callback should not have been called
            assert not on_change.called
        finally:
            manager.stop()
            if ignored_file.exists():
                ignored_file.unlink()
            if ignored_dir.exists():
                ignored_dir.rmdir()
    
    def test_debouncing_batches_rapid_changes(
        self, watchdog_available, test_repo_path
    ):
        """Test that rapid changes are debounced."""
        from sigil_mcp.watcher import FileWatchManager
        
        on_change = Mock()
        manager = FileWatchManager(
            on_change=on_change,
            ignore_dirs=[],
            ignore_extensions=[]
        )
        manager.start()
        
        try:
            manager.watch_repository("test_repo", test_repo_path)
            
            # Make rapid changes to the same file
            test_file = test_repo_path / "main.py"
            for i in range(5):
                content = test_file.read_text()
                test_file.write_text(content + f"\n# Change {i}")
                time.sleep(0.1)  # Small delay between writes
            
            # Wait for debounce + processing
            time.sleep(3.0)
            
            # Should be called only once (or very few times) due to debouncing
            assert on_change.called
            # Exact count depends on timing, but should be much less than 5
            assert on_change.call_count <= 3
        finally:
            manager.stop()


class TestFileWatchingIntegrationWithIndexing:
    """Test file watching integrated with indexing."""
    
    @pytest.fixture
    def watchdog_available(self):
        """Ensure watchdog is available for tests."""
        try:
            import watchdog  # noqa: F401
            return True
        except ImportError:
            pytest.skip("watchdog not installed")
    
    def test_file_change_triggers_reindex(
        self, watchdog_available, indexed_repo, test_repo_path
    ):
        """Test that file changes trigger re-indexing."""
        from sigil_mcp.watcher import FileWatchManager
        
        index = indexed_repo["index"]
        repo_name = indexed_repo["repo_name"]
        
        # Track re-index calls
        reindex_called = []
        
        def on_change(repo_name, file_path, event_type):
            success = index.index_file(repo_name, test_repo_path, file_path)
            reindex_called.append((repo_name, file_path, event_type, success))
        
        manager = FileWatchManager(on_change=on_change)
        manager.start()
        
        try:
            manager.watch_repository(repo_name, test_repo_path)
            
            # Modify a file
            test_file = test_repo_path / "main.py"
            original_content = test_file.read_text()
            test_file.write_text(original_content + "\n\ndef new_function():\n    pass\n")
            
            # Wait for processing
            time.sleep(3.0)
            
            # Check re-index was called
            assert len(reindex_called) > 0
            assert reindex_called[0][0] == repo_name
            assert reindex_called[0][1] == test_file
            assert reindex_called[0][3] is True  # success
            
            # Verify the new function is searchable
            results = index.search_code("new_function", repo=repo_name)
            assert len(results) > 0
        finally:
            manager.stop()
    
    def test_file_deletion_removes_from_index(
        self, watchdog_available, indexed_repo, test_repo_path
    ):
        """Deleting a file via watcher should remove it from the index."""
        from sigil_mcp.watcher import FileWatchManager
        import time

        index = indexed_repo["index"]
        repo_name = indexed_repo["repo_name"]

        # Ensure a known file is present and searchable
        test_file = test_repo_path / "main.py"
        original_content = test_file.read_text()
        unique_marker = "# delete_me_marker"
        test_file.write_text(original_content + f"\n{unique_marker}\n")

        index.index_file(repo_name, test_repo_path, test_file)
        time.sleep(0.5)
        results = index.search_code(unique_marker, repo=repo_name)
        assert any(r.path.endswith("main.py") for r in results)

        # Wire watcher to a local on_change that uses the fixture index directly
        def on_change(repo_name_arg, file_path_arg, event_type_arg):
            if event_type_arg == "deleted":
                index.remove_file(repo_name_arg, test_repo_path, file_path_arg)
            else:
                index.index_file(repo_name_arg, test_repo_path, file_path_arg)

        manager = FileWatchManager(on_change=on_change)
        manager.start()

        try:
            manager.watch_repository(repo_name, test_repo_path)

            # Ensure watcher has started
            time.sleep(0.5)

            # Delete the file on disk
            test_file.unlink()

            # Wait for debounce + processing
            time.sleep(3.5)

            # The marker should no longer be searchable from main.py
            results_after = index.search_code(unique_marker, repo=repo_name)
            assert all(not r.path.endswith("main.py") for r in results_after)
        finally:
            manager.stop()
    
    def test_granular_reindex_faster_than_full(
        self, watchdog_available, indexed_repo, test_repo_path
    ):
        """Test that granular re-indexing is faster than full re-index."""
        index = indexed_repo["index"]
        repo_name = indexed_repo["repo_name"]
        
        # Measure full re-index time
        start = time.time()
        index.index_repository(repo_name, test_repo_path, force=True)
        full_reindex_time = time.time() - start
        
        # Measure granular re-index time for single file
        test_file = test_repo_path / "main.py"
        start = time.time()
        index.index_file(repo_name, test_repo_path, test_file)
        granular_reindex_time = time.time() - start
        
        # Granular should be significantly faster
        assert granular_reindex_time < full_reindex_time / 2
    
    def test_multiple_repo_watching(
        self, watchdog_available, temp_dir, test_index
    ):
        """Test watching multiple repositories simultaneously."""
        from sigil_mcp.watcher import FileWatchManager
        
        # Create two test repos
        repo1 = temp_dir / "repo1"
        repo1.mkdir()
        (repo1 / "file1.py").write_text("def func1(): pass")
        
        repo2 = temp_dir / "repo2"
        repo2.mkdir()
        (repo2 / "file2.py").write_text("def func2(): pass")
        
        # Index both repos
        test_index.index_repository("repo1", repo1)
        test_index.index_repository("repo2", repo2)
        
        changes = []
        
        def on_change(repo_name, file_path, event_type):
            changes.append(repo_name)
        
        manager = FileWatchManager(on_change=on_change)
        manager.start()
        
        try:
            manager.watch_repository("repo1", repo1)
            manager.watch_repository("repo2", repo2)
            
            assert manager.is_watching("repo1")
            assert manager.is_watching("repo2")
            
            # Modify files in both repos
            (repo1 / "file1.py").write_text("def func1(): pass  # modified")
            (repo2 / "file2.py").write_text("def func2(): pass  # modified")
            
            # Wait for processing
            time.sleep(3.0)
            
            # Both repos should have triggered callbacks
            assert "repo1" in changes
            assert "repo2" in changes
        finally:
            manager.stop()


class TestRepositoryWatcher:
    """Test RepositoryWatcher class directly."""
    
    @pytest.fixture
    def watchdog_available(self):
        """Ensure watchdog is available for tests."""
        try:
            import watchdog  # noqa: F401
            return True
        except ImportError:
            pytest.skip("watchdog not installed")
    
    def test_should_ignore_method(self, watchdog_available, test_repo_path):
        """Test the _should_ignore method logic."""
        from sigil_mcp.watcher import RepositoryWatcher
        
        on_change = Mock()
        watcher = RepositoryWatcher(
            repo_name="test",
            repo_path=test_repo_path,
            on_change=on_change,
            ignore_dirs=[".git", "__pycache__"],
            ignore_extensions=[".pyc", ".so"]
        )
        
        # Should ignore files with ignored extensions
        assert watcher._should_ignore(test_repo_path / "test.pyc")
        assert watcher._should_ignore(test_repo_path / "lib.so")
        
        # Should ignore files in ignored directories
        assert watcher._should_ignore(test_repo_path / ".git" / "config")
        assert watcher._should_ignore(test_repo_path / "__pycache__" / "test.pyc")
        
        # Should not ignore normal files
        assert not watcher._should_ignore(test_repo_path / "main.py")
        assert not watcher._should_ignore(test_repo_path / "lib" / "helper.py")
    
    def test_schedule_change_debouncing(self, watchdog_available, test_repo_path):
        """Test that _schedule_change properly debounces."""
        from sigil_mcp.watcher import RepositoryWatcher
        
        on_change = Mock()
        watcher = RepositoryWatcher(
            repo_name="test",
            repo_path=test_repo_path,
            on_change=on_change,
            debounce_seconds=2.0
        )
        
        test_file = test_repo_path / "main.py"
        
        # Schedule same file multiple times rapidly
        for i in range(5):
            watcher._schedule_change(str(test_file), "modified")
            time.sleep(0.1)
        
        # Check pending changes - should only have one entry
        assert len(watcher.pending_changes) == 1
        assert str(test_file) in watcher.pending_changes
        
        watcher.stop()
    
    def test_processing_thread_processes_changes(
        self, watchdog_available, test_repo_path
    ):
        """Test that the processing thread processes scheduled changes."""
        from sigil_mcp.watcher import RepositoryWatcher
        
        on_change = Mock()
        watcher = RepositoryWatcher(
            repo_name="test",
            repo_path=test_repo_path,
            on_change=on_change,
            debounce_seconds=1.0
        )
        
        test_file = test_repo_path / "main.py"
        watcher._schedule_change(str(test_file), "modified")
        
        # Wait for debounce + processing
        time.sleep(2.0)
        
        # Callback should have been called
        assert on_change.called
        
        watcher.stop()
