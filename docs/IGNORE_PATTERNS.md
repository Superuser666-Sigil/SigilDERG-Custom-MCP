# Index Ignore Patterns

This document describes how `index.ignore_patterns` are used by the Sigil indexer and how to override them per-repository.

Overview
--------
- `index.ignore_patterns` in `config.json` is a global, project-level list of shell-style glob patterns the indexer will use to skip files and directories during indexing.
- Patterns support simple globbing (e.g. `*.pyc`), directory suffixes (e.g. `.git/`), and negation/allow rules using a leading `!` (e.g. `!/sigil-admin-ui/src/lib/`).

Precedence
----------
When deciding whether to skip a path the indexer evaluates patterns in this order (highest to lowest precedence):

1. Per-repo allow (negation) patterns (explicit `!pattern` defined on the repository entry)
2. Per-repo ignore patterns (defined on the repository entry)
3. Global allow (negation) patterns (explicit `!pattern` in `index.ignore_patterns`)
4. Global ignore patterns (entries in `index.ignore_patterns`)
5. Built-in heuristics and size/extension-based checks (legacy rules)

This ordering allows repository-specific whitelists to re-include paths that would otherwise be ignored globally.

Per-repo overrides
------------------
You can configure per-repo settings in the top-level `repositories` mapping in `config.json` in one of two forms:

1. Simple form (default behavior):

```json
"repositories": {
  "sigil": "/home/dave/dev/SigilDERG-Custom-MCP"
}
```

2. Expanded form (with per-repo options):

```json
"repositories": {
  "sigil": {
    "path": "/home/dave/dev/SigilDERG-Custom-MCP",
    "respect_gitignore": true,
    "ignore_patterns": [
      "target/",
      "*.rlib",
      "!/sigil-admin-ui/src/lib/**"
    ]
  }
}
```

- `respect_gitignore` (bool): when true the repo will honor `.gitignore` files where applicable. (Existing behavior may vary; use this flag where you want to align with repository-managed ignores.)
- `ignore_patterns` (list): additional per-repo glob patterns or negations to apply with higher precedence than the global patterns.

Notes and examples
------------------
- Patterns are evaluated against the file's path relative to the repository root when available, then against the full path and file basename.
- Negation patterns begin with `!` and re-include matches from earlier ignore rules.
- Lines beginning with `#` in `index.ignore_patterns` are treated as comments and ignored.

Examples
--------
- Ignore all compiled Python files globally:

```json
"index": {
  "ignore_patterns": ["*.pyc"]
}
```

- Re-include a particular folder from being ignored globally:

```json
"index": {
  "ignore_patterns": ["node_modules/", "!/sigil-admin-ui/src/lib/**"]
}
```

- Per-repo override example (re-include a specific library for admin UI):

```json
"repositories": {
  "sigil-admin-ui": {
    "path": "/home/dave/dev/SigilDERG-Custom-MCP/sigil-admin-ui",
    "ignore_patterns": ["!/src/lib/**"]
  }
}
```

Implementation details
----------------------
- The indexer consults the per-repository `ignore_patterns` (if provided) first, then the global `index.ignore_patterns`, and finally falls back to legacy heuristics such as extension-based skips and file-size thresholds.
- Glob matching uses shell-style wildcards and is intentionally permissive to match common `.gitignore`-style expressions.

If you'd like, I can also add code to parse and honor repository `.gitignore` files (respecting `.gitignore` semantics) as an additional option; tell me if you'd like that behavior and I'll implement it.