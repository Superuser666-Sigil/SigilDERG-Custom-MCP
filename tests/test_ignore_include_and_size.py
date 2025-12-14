from pathlib import Path

from sigil_mcp.ignore_utils import should_ignore


def test_include_overrides_gitignore(tmp_path: Path):
    # repo layout: a/b/c.txt
    repo = tmp_path
    a = repo / "a" / "b"
    a.mkdir(parents=True)
    target = a / "c.txt"
    target.write_text("hello")

    # gitignore would ignore *.txt, but include_patterns explicitly includes a/b/c.txt
    gitignore_patterns = ["*.txt"]
    include_patterns = ["a/b/c.txt"]

    assert should_ignore(target, repo, include_patterns=include_patterns, gitignore_patterns=gitignore_patterns) is False


def test_large_file_ignored_by_size(tmp_path: Path):
    repo = tmp_path
    f = repo / "large.bin"
    # create a file larger than 1MB
    data = b"0" * (1_200_000)
    f.write_bytes(data)

    # Default max_size_bytes in should_ignore is 1_000_000, so this should be ignored
    assert should_ignore(f, repo) is True

    # If we increase threshold, it should no longer be ignored
    assert should_ignore(f, repo, max_size_bytes=2_000_000) is False
