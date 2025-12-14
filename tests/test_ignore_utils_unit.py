# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

from pathlib import Path

from sigil_mcp.ignore_utils import (
    load_gitignore,
    load_include_patterns,
    _match_pattern,
    is_ignored_by_gitignore,
    should_ignore,
)


def test_load_gitignore_reads_gitignore_and_info_exclude(tmp_path: Path):
    repo = tmp_path
    gitignore = repo / ".gitignore"
    gitignore.write_text("# comment\nfoo/\n*.log\n!keep.log\n")
    info_dir = repo / ".git" / "info"
    info_dir.mkdir(parents=True)
    (info_dir / "exclude").write_text("bar/\n# skip\n*.tmp\n")

    patterns = load_gitignore(repo)

    # Order is preserved and combines both files
    assert patterns == ["foo/", "*.log", "!keep.log", "bar/", "*.tmp"]


def test_load_include_patterns_reads_both_filenames(tmp_path: Path):
    repo = tmp_path
    (repo / ".sigil_mcp_include").write_text("# c1\nfoo.txt\n")
    (repo / ".sigil_index_include").write_text("bar/**\n\n")

    patterns = load_include_patterns(repo)

    assert patterns == ["foo.txt", "bar/**"]


def test_match_pattern_variants():
    assert _match_pattern("src/foo/bar.txt", "src/**")
    assert _match_pattern("src/foo/bar.txt", "/src/foo/*")
    assert _match_pattern("src/foo/bar.txt", "bar.txt")
    assert _match_pattern("foo/bar.txt", "foo/")
    assert not _match_pattern("src/foo/bar.txt", "baz/")


def test_is_ignored_by_gitignore_honors_negation(tmp_path: Path):
    repo = tmp_path
    target = repo / "keep.txt"
    target.write_text("ok")

    patterns = ["*.txt", "!keep.txt"]
    assert is_ignored_by_gitignore(target, repo, patterns) is False


def test_should_ignore_precedence_and_heuristics(tmp_path: Path):
    repo = tmp_path
    repo.mkdir(exist_ok=True)
    keep_file = repo / "keep.py"
    keep_file.write_text("print('hi')")
    ignored_dir = repo / "build"
    ignored_dir.mkdir()
    ignored_file = ignored_dir / "out.bin"
    ignored_file.write_bytes(b"binary")

    # Include pattern should override repo/global ignores
    include_patterns = ["keep.py"]
    repo_ignore_patterns = ["*.py"]  # would normally ignore keep.py

    assert should_ignore(
        keep_file,
        repo_root=repo,
        include_patterns=include_patterns,
        repo_ignore_patterns=repo_ignore_patterns,
    ) is False

    # Directory heuristic should ignore child paths
    assert should_ignore(
        ignored_file,
        repo_root=repo,
        ignore_dirs=["build"],
    ) is True

    # Hidden file heuristic
    hidden = repo / ".secret"
    hidden.write_text("x")
    assert should_ignore(hidden, repo_root=repo) is True

    # Size heuristic
    large = repo / "large.bin"
    large.write_bytes(b"0" * (2_000_000))
    assert should_ignore(large, repo_root=repo, max_size_bytes=1_000) is True


def test_load_gitignore_handles_errors(monkeypatch, tmp_path: Path):
    repo = tmp_path
    gitignore = repo / ".gitignore"
    gitignore.write_text("*.py")

    def fake_read_text(self, *a, **k):
        raise OSError("boom")

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    assert load_gitignore(repo) == []


def test_load_include_patterns_handles_errors(monkeypatch, tmp_path: Path):
    repo = tmp_path
    (repo / ".sigil_mcp_include").write_text("foo.txt")

    def fake_read_text(self, *a, **k):
        raise OSError("nope")

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    assert load_include_patterns(repo) == []


def test_is_ignored_by_gitignore_edge_cases(tmp_path: Path):
    repo = tmp_path
    outside = Path("/tmp/outside.txt")
    assert is_ignored_by_gitignore(outside, repo, ["*.txt"]) is False
    assert is_ignored_by_gitignore(repo / "file.txt", repo, []) is False
    assert is_ignored_by_gitignore(repo / "file.txt", repo, ["", "  "]) is False


def test_match_pattern_unanchored_wildcard():
    assert _match_pattern("a/b/c.txt", "c.txt") is True


def test_should_ignore_includes_and_whitelists(tmp_path: Path):
    repo = tmp_path
    target = repo / "keep.md"
    target.write_text("x")
    includes = ["keep.md"]
    repo_patterns = ["*.md"]
    assert should_ignore(
        target,
        repo_root=repo,
        include_patterns=includes,
        repo_ignore_patterns=repo_patterns,
    ) is False


def test_should_ignore_pattern_lists_and_heuristics(tmp_path: Path, monkeypatch):
    repo = tmp_path
    nested = repo / "ignored" / "sub"
    nested.mkdir(parents=True)
    target = nested / "file.log"
    target.write_text("x")
    # Trigger repo ignore patterns
    assert should_ignore(
        target,
        repo_root=repo,
        repo_ignore_patterns=["*.log"],
    ) is True

    allowed = repo / "allowed.txt"
    allowed.write_text("x")
    assert should_ignore(
        allowed,
        repo_root=repo,
        repo_ignore_patterns=["!allowed.txt"],
        config_ignore_patterns=["*.txt"],
    ) is False

    cargo = Path("/tmp/CARGO_TARGET_CACHE/file")
    assert should_ignore(cargo, repo_root=repo) is True

    backup = repo / "notes.backup"
    backup.write_text("x")
    assert should_ignore(backup, repo_root=repo) is True

    ext = repo / "skip.tmp"
    ext.write_text("x")
    assert should_ignore(ext, repo_root=repo, ignore_extensions=[".tmp"]) is True

    timestamped = repo / ".timestamp-123.txt"
    timestamped.write_text("x")
    assert should_ignore(timestamped, repo_root=repo) is True

    broken = repo / "broken.txt"

    class BadPath(Path):
        _flavour = Path(".")._flavour

        def stat(self):
            raise OSError("no stat")

    bad = BadPath(str(broken))
    assert should_ignore(bad, repo_root=repo) is True

    # gitignore patterns match when provided
    gitignore_patterns = ["*.cfg"]
    cfg_file = repo / "settings.cfg"
    cfg_file.write_text("x")
    assert should_ignore(cfg_file, repo_root=repo, gitignore_patterns=gitignore_patterns) is True


def test_load_gitignore_handles_info_exclude_error(monkeypatch, tmp_path: Path):
    repo = tmp_path
    (repo / ".gitignore").write_text("foo/\n")
    info_dir = repo / ".git" / "info"
    info_dir.mkdir(parents=True)
    exclude_file = info_dir / "exclude"
    exclude_file.write_text("bar/")

    original_read = Path.read_text

    def read_text(path, *a, **k):
        if Path(path) == exclude_file:
            raise OSError("fail")
        return original_read(path, *a, **k)

    monkeypatch.setattr(Path, "read_text", read_text)
    patterns = load_gitignore(repo)
    assert "foo/" in patterns


def test_should_ignore_when_loader_raises(monkeypatch, tmp_path: Path):
    repo = tmp_path
    target = repo / "file.txt"
    target.write_text("x")

    monkeypatch.setattr(
        "sigil_mcp.ignore_utils.load_include_patterns",
        lambda root: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "sigil_mcp.ignore_utils.load_gitignore",
        lambda root: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert should_ignore(target, repo_root=repo, include_patterns=None, gitignore_patterns=None) is False
