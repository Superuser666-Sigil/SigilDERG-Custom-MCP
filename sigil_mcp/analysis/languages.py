"""Language and file classification helpers."""

from __future__ import annotations

import re
from pathlib import Path

CODE_LIKE_EXTS = {
    ".py",
    ".pyw",
    ".pyi",
    ".pyx",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".rs",
    ".go",
    ".java",
    ".kt",
    ".kts",
    ".cs",
    ".cpp",
    ".cxx",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".m",
    ".mm",
    ".swift",
    ".scala",
    ".rb",
    ".php",
    ".pl",
    ".pm",
    ".sql",
    ".psql",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".psm1",
    ".lua",
    ".hs",
    ".ml",
    ".ex",
    ".exs",
    ".dart",
    ".groovy",
    ".r",
    ".jl",
    ".jsonnet",
    ".cue",
}

DOC_LIKE_EXTS = {
    ".md",
    ".mdx",
    ".rst",
    ".adoc",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".csv",
    ".tsv",
    ".log",
    ".pdf",
    ".doc",
    ".docx",
    ".rtf",
}

CONFIG_LIKE_EXTS = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"}
DATA_LIKE_EXTS = {".csv", ".tsv", ".jsonl", ".ndjson", ".parquet"}

EXT_LANGUAGE_MAP = {
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".go": "go",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".m": "objective-c",
    ".mm": "objective-cpp",
    ".swift": "swift",
    ".scala": "scala",
    ".rb": "ruby",
    ".php": "php",
    ".pl": "perl",
    ".pm": "perl",
    ".sql": "sql",
    ".psql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".lua": "lua",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".ex": "elixir",
    ".exs": "elixir",
    ".dart": "dart",
    ".groovy": "groovy",
    ".r": "r",
    ".jl": "julia",
    ".jsonnet": "jsonnet",
    ".cue": "cue",
}


def classify_path(rel_path: str, sample_text: str | None = None) -> dict[str, object]:
    """Heuristic classification for semantic search weighting and metadata."""
    ext = Path(rel_path).suffix.lower()
    is_doc = ext in DOC_LIKE_EXTS
    is_config = ext in CONFIG_LIKE_EXTS
    is_data = ext in DATA_LIKE_EXTS
    is_code = ext in CODE_LIKE_EXTS and not (is_doc or is_config or is_data)
    language = EXT_LANGUAGE_MAP.get(ext)

    text = (sample_text or "").strip()
    if ext == ".h":
        if re.search(r"@interface|@implementation|@class\b", text):
            language = "objective-c"
            is_code = True
        elif re.search(r"\bnamespace\b|\bstd::|\btemplate\s*<", text):
            language = "cpp"
            is_code = True
        else:
            language = language or "c"
    elif ext in {".ts", ".tsx"}:
        language = "typescript"
        is_code = True
    elif ext in {".py", ".pyi", ".pyw"}:
        language = "python"
        is_code = True
    elif ext in {".m", ".mm"}:
        if re.search(r"@interface|@implementation|@class\b", text):
            language = "objective-c" if ext == ".m" else "objective-cpp"
            is_code = True
    elif not language:
        language = "unknown"

    artifact_dirs = {
        "egg-info",
        "dist-info",
        "site-packages",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".cache",
        "build",
        "dist",
        "venv",
        ".venv",
    }
    filename = Path(rel_path).name.lower()
    if any(part in artifact_dirs for part in Path(rel_path).parts):
        is_code = False
        is_doc = False
        is_config = False
        is_data = True

    metadata_files = {
        "pkg-info",
        "metadata",
        "record",
        "wheel",
        "license",
        "license.txt",
        "copying",
        "authors",
        "install",
        "changelog",
        "readme",
    }
    if filename in metadata_files:
        is_code = False
        is_doc = True
        is_config = False
        is_data = False

    if not ext:
        is_code = False
        language = None
        if text:
            if re.search(r"^#!", text):
                is_code = True
            elif re.search(
                r"\b(class|def|function|package|namespace|public\s+static\s+void)\b", text
            ):
                is_code = True
            elif text.count(";") > 3 and text.count("{") + text.count("}") > 1:
                is_code = True

    if not is_code:
        language = language or "unknown"
    return {
        "is_code": is_code,
        "is_doc": is_doc,
        "is_config": is_config,
        "is_data": is_data,
        "extension": ext or None,
        "language": language,
    }
