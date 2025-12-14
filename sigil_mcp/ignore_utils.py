import fnmatch
from collections.abc import Iterable
from pathlib import Path


def load_gitignore(repo_root: Path) -> list[str]:
    """Load .gitignore-style patterns from repo root (simple support).

    Returns a list of patterns in order. Supports negation (!) lines.
    """
    patterns: list[str] = []
    try:
        gitignore = repo_root / '.gitignore'
        if gitignore.exists():
            for line in gitignore.read_text(encoding='utf-8', errors='ignore').splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                patterns.append(line)
    except Exception:
        # On any error, return empty list (fail-open)
        return []

    # Also consider .git/info/exclude if present
    try:
        info_exclude = repo_root / '.git' / 'info' / 'exclude'
        if info_exclude.exists():
            for line in info_exclude.read_text(encoding='utf-8', errors='ignore').splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                patterns.append(line)
    except Exception:
        pass

    return patterns


def load_include_patterns(repo_root: Path) -> list[str]:
    """Load explicit include patterns from repository to override .gitignore.

    Supported filenames (checked in order):
      - .sigil_mcp_include
      - .sigil_index_include

    Each line is a pattern similar to .gitignore; lines starting with '#' are ignored.
    Patterns here act as explicit includes: if a path matches any include pattern,
    it will be considered for indexing/watching even if .gitignore would ignore it.
    """
    patterns: list[str] = []
    try:
        for fname in ('.sigil_mcp_include', '.sigil_index_include'):
            fpath = repo_root / fname
            if fpath.exists():
                for line in fpath.read_text(encoding='utf-8', errors='ignore').splitlines():
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    patterns.append(line)
    except Exception:
        return []

    return patterns


def _match_pattern(rel_path: str, pattern: str) -> bool:
    """Match a single gitignore-style pattern against a repo-relative path.

    This is a conservative, lightweight implementation that supports:
    - anchored patterns starting with '/'
    - wildcard '*' and '**' via fnmatch
    - directory patterns ending with '/'
    - basename matches
    It does not implement every gitignore nuance but is sufficient for typical use.
    """
    if pattern.endswith('/'):
        # directory pattern
        p = pattern.rstrip('/')
        # match if rel_path equals p or is inside p/
        if rel_path == p or rel_path.startswith(p + '/'):
            return True
        # also allow matching with glob
        return fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(rel_path, f"**/{p}")

    anchored = pattern.startswith('/')
    pat = pattern.lstrip('/')

    # direct match
    if fnmatch.fnmatch(rel_path, pat):
        return True

    # basename match
    if fnmatch.fnmatch(Path(rel_path).name, pat):
        return True

    # unanchored patterns should match anywhere in the tree
    if not anchored:
        if fnmatch.fnmatch(rel_path, f"**/{pat}"):
            return True

    return False


def is_ignored_by_gitignore(path: Path, repo_root: Path, patterns: list[str]) -> bool:
    """Return True if path is ignored according to provided gitignore patterns.

    Patterns are processed in order; later negation (!pattern) overrides previous matches.
    """
    if not patterns:
        return False

    try:
        rel = str(path.relative_to(repo_root).as_posix())
    except Exception:
        # If path is outside repo, do not ignore here
        return False

    ignored = False
    for pat in patterns:
        if not pat:
            continue
        if pat.startswith('!'):
            sub = pat[1:]
            if _match_pattern(rel, sub):
                ignored = False
        else:
            if _match_pattern(rel, pat):
                ignored = True

    return ignored


def should_ignore(
    path: Path,
    repo_root: Path | None = None,
    *,
    config_ignore_patterns: list[str] | None = None,
    repo_ignore_patterns: list[str] | None = None,
    include_patterns: list[str] | None = None,
    gitignore_patterns: list[str] | None = None,
    ignore_dirs: Iterable[str] | None = None,
    ignore_extensions: Iterable[str] | None = None,
    max_size_bytes: int = 1_000_000,
) -> bool:
    """Unified ignore decision used by both watcher and indexer.

    Precedence (highest first):
      - explicit include patterns (repo-level include files)
      - per-repo allow/ignore patterns (`repo_ignore_patterns`, `!` negations)
      - global allow/ignore patterns (`config_ignore_patterns`)
      - legacy heuristics (ignore_dirs, ignore_extensions, hidden files, size)
      - repo .gitignore patterns (loaded via `load_gitignore`) as a final check

    The function is permissive and fail-open on errors.
    """
    try:
        rel = None
        if repo_root is not None:
            try:
                rel = str(path.relative_to(repo_root).as_posix())
            except Exception:
                rel = None

        # If include_patterns not provided, try loading repo include files
        if include_patterns is None and repo_root is not None:
            try:
                include_patterns = load_include_patterns(repo_root)
            except Exception:
                include_patterns = []

        # If gitignore_patterns not provided, try loading .gitignore
        if gitignore_patterns is None and repo_root is not None:
            try:
                gitignore_patterns = load_gitignore(repo_root)
            except Exception:
                gitignore_patterns = []

        # Explicit includes (if match, never ignore)
        if include_patterns:
            try:
                if is_ignored_by_gitignore(path, repo_root or Path('.'), include_patterns):
                    return False
            except Exception:
                pass

        # Helper to split patterns into allows (negations) and positives
        def _split_patterns(pats: list[str] | None):
            allows: list[str] = []
            pos: list[str] = []
            if not pats:
                return allows, pos
            for p in pats:
                if not p or not isinstance(p, str):
                    continue
                s = p.strip()
                if not s or s.startswith('#'):
                    continue
                if s.startswith('!'):
                    allows.append(s[1:].strip())
                else:
                    pos.append(s)
            return allows, pos

        repo_allows, repo_pos = _split_patterns(repo_ignore_patterns)
        global_allows, global_pos = _split_patterns(config_ignore_patterns)

        # Matching helper using fnmatch against rel, full path, and basename
        def _match_any(patterns: list[str]) -> bool:
            for pat in patterns:
                try:
                    if rel:
                        if _match_pattern(rel, pat):
                            return True
                except Exception:
                    pass
                try:
                    if fnmatch.fnmatch(path.as_posix(), pat):
                        return True
                except Exception:
                    pass
                try:
                    if fnmatch.fnmatch(path.name, pat):
                        return True
                except Exception:
                    pass
            return False

        # Precedence: repo allows -> repo ignores -> global allows -> global ignores
        if repo_allows and _match_any(repo_allows):
            return False
        if repo_pos and _match_any(repo_pos):
            return True
        if global_allows and _match_any(global_allows):
            return False
        if global_pos and _match_any(global_pos):
            return True

        # Legacy heuristics: ignore dirs, extensions, hidden files, timestamp tokens, backup names
        skip_dirs = set(ignore_dirs or [])
        skip_exts = set(ignore_extensions or [])

        # Check if any parent directory matches configured ignore dirs
        for parent in path.parents:
            if parent.name in skip_dirs or ('.' + parent.name) in skip_dirs:
                return True

        # Cargo target cache special-case
        try:
            s = str(path).lower()
            if "cargo_target_cache" in s or "/cargo_target_cache/" in s:
                return True
        except Exception:
            pass

        # Backup/timestamped files
        try:
            name_l = path.name.lower()
            if ".backup" in name_l or name_l.endswith("~") or name_l.endswith(".bak"):
                return True
        except Exception:
            pass

        # Check extension
        try:
            if path.suffix and path.suffix.lower() in skip_exts:
                return True
        except Exception:
            pass

        # Hidden files/dirs
        if any(part.startswith('.') for part in path.parts):
            return True

        # Vite timestamp filenames
        if '.timestamp-' in path.name:
            return True

        # Large files
        try:
            if path.stat().st_size > max_size_bytes:
                return True
        except Exception:
            # If we can't stat, be conservative and skip
            return True

        # Finally, consult .gitignore patterns if available
        try:
            if gitignore_patterns and repo_root:
                if is_ignored_by_gitignore(path, repo_root, gitignore_patterns):
                    return True
        except Exception:
            pass

    except Exception:
        # Fail-open on any unexpected error
        return False

    return False
