import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from sigil_mcp import watcher as watcher_mod


def test_repository_watcher_processes_batched_changes(tmp_path):
    repo_path = tmp_path
    target = repo_path / "file.txt"
    target.write_text("hello")

    calls: list[tuple[str, Path, str]] = []

    watcher = watcher_mod.RepositoryWatcher(
        repo_name="repo",
        repo_path=repo_path,
        on_change=lambda repo, path, event: calls.append((repo, path, event)),
        debounce_seconds=0.05,
        honor_gitignore=False,
    )

    try:
        watcher._schedule_change(str(target), "modified")
        deadline = time.time() + 2.0
        while not calls and time.time() < deadline:
            time.sleep(0.05)

        assert calls, "Expected scheduled change to be processed"
        repo, path, event = calls[0]
        assert repo == "repo"
        assert path == target.resolve()
        assert event == "modified"
    finally:
        watcher.stop()


def test_repository_watcher_handles_gitignore_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(watcher_mod, "load_gitignore", lambda *_, **__: (_ for _ in ()).throw(RuntimeError("fail")))
    monkeypatch.setattr(watcher_mod, "load_include_patterns", lambda *_, **__: (_ for _ in ()).throw(RuntimeError("fail")))
    watcher = watcher_mod.RepositoryWatcher(
        repo_name="repo",
        repo_path=tmp_path,
        on_change=lambda *_: None,
        debounce_seconds=0.01,
        honor_gitignore=True,
    )
    try:
        assert watcher._gitignore_patterns == []
        assert watcher._include_patterns == []
    finally:
        watcher.stop()


def test_repository_watcher_should_ignore_get_config_error(monkeypatch, tmp_path):
    monkeypatch.setattr(watcher_mod, "get_config", lambda: (_ for _ in ()).throw(RuntimeError("cfg")))
    watcher = watcher_mod.RepositoryWatcher(
        repo_name="repo",
        repo_path=tmp_path,
        on_change=lambda *_: None,
        debounce_seconds=0.01,
        honor_gitignore=False,
    )
    try:
        path = tmp_path / "a.txt"
        path.write_text("x")
        assert watcher._should_ignore(path) is False
    finally:
        watcher.stop()


def test_repository_watcher_event_handlers(monkeypatch, tmp_path):
    calls = []
    watcher = watcher_mod.RepositoryWatcher(
        repo_name="repo",
        repo_path=tmp_path,
        on_change=lambda *args: calls.append(args),
        debounce_seconds=0.01,
        honor_gitignore=False,
    )
    watcher._should_ignore_path = lambda *_: False  # type: ignore[assignment]
    watcher._schedule_change = lambda path, event: calls.append((path, event))  # type: ignore[assignment]

    class Ev:
        def __init__(self, path):
            self.src_path = str(path)
            self.is_directory = False

    try:
        watcher.on_modified(Ev(tmp_path / "m.txt"))
        watcher.on_created(Ev(tmp_path / "c.txt"))
        watcher.on_deleted(Ev(tmp_path / "d.txt"))
        assert ("m.txt" not in str(calls[0])) is False  # ensure calls populated
    finally:
        watcher.stop()


def test_repository_watcher_event_handlers_ignored(monkeypatch, tmp_path):
    watcher = watcher_mod.RepositoryWatcher(
        repo_name="repo",
        repo_path=tmp_path,
        on_change=lambda *_: None,
        debounce_seconds=0.01,
        honor_gitignore=False,
    )
    watcher._should_ignore_path = lambda *_: True  # type: ignore[assignment]
    watcher._schedule_change = lambda *_: (_ for _ in ()).throw(RuntimeError("should not run"))  # type: ignore[assignment]

    class Ev:
        def __init__(self, path):
            self.src_path = str(path)
            self.is_directory = False

    try:
        watcher.on_modified(Ev(tmp_path / "m.txt"))
        watcher.on_created(Ev(tmp_path / "c.txt"))
        watcher.on_deleted(Ev(tmp_path / "d.txt"))
    finally:
        watcher.stop()


def test_repository_watcher_ignore_logic(tmp_path):
    repo_path = tmp_path
    (repo_path / ".git").mkdir()
    watched = watcher_mod.RepositoryWatcher(
        repo_name="repo",
        repo_path=repo_path,
        on_change=lambda *_: None,
        debounce_seconds=0.01,
        ignore_dirs=[".git"],
        honor_gitignore=False,
    )
    try:
        assert watched._should_ignore_path(str(repo_path / ".git/config")) is True
        assert watched._should_ignore_path(str(repo_path / "src/app.py")) is False
    finally:
        watched.stop()


def test_file_watch_manager_without_real_watchdog(monkeypatch, tmp_path):
    scheduled: list[tuple[object, str, bool]] = []

    class DummyObserver:
        def __init__(self):
            self.started = False
            self.stopped = False
            self.join_timeout = None

        def schedule(self, watcher, path, recursive=True):
            scheduled.append((watcher, path, recursive))

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def join(self, timeout=None):
            self.join_timeout = timeout

    monkeypatch.setattr(watcher_mod, "Observer", DummyObserver)
    monkeypatch.setattr(watcher_mod, "WATCHDOG_AVAILABLE", True)

    manager = watcher_mod.FileWatchManager(on_change=Mock())
    manager.start()
    manager.watch_repository("demo", tmp_path, honor_gitignore=False)
    # Duplicate watch should hit already-watching branch
    manager.watch_repository("demo", tmp_path, honor_gitignore=False)

    try:
        assert manager.is_watching("demo") is True
        assert isinstance(manager.observer, DummyObserver)
        assert scheduled and scheduled[0][1] == str(tmp_path)
        manager.unwatch_repository("demo")
    finally:
        manager.stop()


def test_file_watch_manager_handles_schedule_errors(monkeypatch, tmp_path):
    class FailingObserver:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

        def schedule(self, watcher, path, recursive=True):
            raise RuntimeError("boom")

        def stop(self):
            pass

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(watcher_mod, "Observer", FailingObserver)
    monkeypatch.setattr(watcher_mod, "WATCHDOG_AVAILABLE", True)

    manager = watcher_mod.FileWatchManager(on_change=Mock())
    manager.start()
    manager.watch_repository("bad", tmp_path, honor_gitignore=False)
    assert manager.is_watching("bad") is False
    manager.stop()


def test_file_watch_manager_server_repo_update_failure(monkeypatch, tmp_path):
    class DummyObserver:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

        def schedule(self, watcher, path, recursive=True):
            pass

        def stop(self):
            pass

        def join(self, timeout=None):
            pass

    class BadServer:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    bad_server = BadServer()
    monkeypatch.setattr("sigil_mcp.server", bad_server, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "sigil_mcp.server", bad_server)
    monkeypatch.setattr(watcher_mod, "Observer", DummyObserver)
    monkeypatch.setattr(watcher_mod, "WATCHDOG_AVAILABLE", True)

    manager = watcher_mod.FileWatchManager(on_change=Mock())
    manager.start()
    manager.watch_repository("repo", tmp_path, honor_gitignore=False)
    manager.stop()


def test_repository_watcher_honor_gitignore_and_errors(monkeypatch, tmp_path):
    # Create patterns to exercise include/gitignore loading
    (tmp_path / ".gitignore").write_text("ignore.me\n")
    (tmp_path / ".sigil_mcp_include").write_text("include.txt\n")

    # Force config lookup to fail to cover fallback path
    monkeypatch.setattr(watcher_mod, "get_config", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    on_change = Mock()
    watcher = watcher_mod.RepositoryWatcher(
        repo_name="repo",
        repo_path=tmp_path,
        on_change=on_change,
        debounce_seconds=0.01,
        honor_gitignore=True,
    )

    try:
        include_file = tmp_path / "include.txt"
        include_file.write_text("hi")
        assert watcher._should_ignore_path(str(include_file)) is False

        ignored_file = tmp_path / "ignore.me"
        ignored_file.write_text("bye")
        assert watcher._should_ignore_path(str(ignored_file)) is True

        # Outside repo should return early
        watcher._schedule_change(str(tmp_path.parent / "other" / "file.txt"), "modified")
    finally:
        watcher.stop()


def test_watcher_handles_processing_errors(monkeypatch, tmp_path):
    triggered = Mock(side_effect=RuntimeError("fail"))
    watcher = watcher_mod.RepositoryWatcher(
        repo_name="repo",
        repo_path=tmp_path,
        on_change=triggered,
        debounce_seconds=0.01,
        honor_gitignore=False,
    )

    try:
        bad_path = tmp_path / "missing.txt"
        watcher._schedule_change(str(bad_path), "deleted")
        time.sleep(0.05)
    finally:
        watcher.stop()

    # Force _schedule_change exception path
    monkeypatch.setattr(watcher_mod.Path, "resolve", lambda self: (_ for _ in ()).throw(RuntimeError("resolve error")))
    watcher_err = watcher_mod.RepositoryWatcher(
        repo_name="repo",
        repo_path=tmp_path,
        on_change=lambda *_: None,
        debounce_seconds=0.01,
        honor_gitignore=False,
    )
    try:
        watcher_err._schedule_change(str(tmp_path / "x.txt"), "modified")
    finally:
        watcher_err.stop()


def test_watcher_imports_with_watchdog_stub(monkeypatch):
    import importlib
    import importlib.util
    import sys
    import types

    fake_root = Path(watcher_mod.__file__).resolve()
    spec = importlib.util.spec_from_file_location("sigil_mcp.watcher_fake", fake_root)

    # Stub watchdog modules so the import path sets WATCHDOG_AVAILABLE = True
    monkeypatch.setitem(sys.modules, "watchdog", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "watchdog.observers",
        types.SimpleNamespace(Observer=type("DummyObserver", (), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "watchdog.events",
        types.SimpleNamespace(FileSystemEventHandler=object, FileSystemEvent=object),
    )

    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    assert module.WATCHDOG_AVAILABLE is True


def test_should_ignore_path_exception(monkeypatch, tmp_path):
    real_Path = watcher_mod.Path

    class BoomPath(real_Path):
        def __new__(cls, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(watcher_mod, "Path", BoomPath)
    watcher = watcher_mod.RepositoryWatcher(
        repo_name="repo",
        repo_path=tmp_path,
        on_change=lambda *_: None,
        debounce_seconds=0.01,
        honor_gitignore=False,
    )
    try:
        assert watcher._should_ignore_path(str(tmp_path / "foo")) is False
    finally:
        watcher.stop()
        monkeypatch.setattr(watcher_mod, "Path", real_Path)


def test_should_ignore_path_ignore_dirs_branch(tmp_path):
    watcher = watcher_mod.RepositoryWatcher(
        repo_name="repo",
        repo_path=tmp_path,
        on_change=lambda *_: None,
        debounce_seconds=0.01,
        honor_gitignore=False,
        ignore_dirs=[".git"],
    )
    try:
        assert watcher._should_ignore_path(str(tmp_path / ".git" / "config")) is True
    finally:
        watcher.stop()


def test_should_ignore_path_ignore_dirs_without_dot(tmp_path):
    watcher = watcher_mod.RepositoryWatcher(
        repo_name="repo",
        repo_path=tmp_path,
        on_change=lambda *_: None,
        debounce_seconds=0.01,
        honor_gitignore=False,
        ignore_dirs=["tmpdir"],
    )
    try:
        assert watcher._should_ignore_path(str(tmp_path / "tmpdir" / "file.txt")) is True
    finally:
        watcher.stop()
