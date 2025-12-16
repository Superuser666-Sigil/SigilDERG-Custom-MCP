"""Stateless text parsing and chunking utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from sigil_mcp.config import get_config

try:
    import tiktoken  # type: ignore

    TIKTOKEN_AVAILABLE = True
except Exception:
    tiktoken = None  # type: ignore
    TIKTOKEN_AVAILABLE = False


def is_jsonl_path(p: str | Path) -> bool:
    """Detect whether a path refers to a JSONL file, handling multi-suffix names."""
    try:
        p0 = Path(p)
        if any(s.lower() == ".jsonl" for s in p0.suffixes):
            return True
        s = str(p).lower()
        if s.endswith(".jsonl"):
            return True
    except Exception:
        pass
    return False


def count_tokens(
    s: str, embed_model: str | None = None, embedding_provider: str | None = None
) -> int:
    """Count tokens for a string using tiktoken when available."""
    if (embedding_provider or "").lower() == "llamacpp":
        try:
            return max(1, int((len(s) + 3) / 4))
        except Exception:
            return max(1, len(s.split()))

    try:
        if TIKTOKEN_AVAILABLE and embed_model:
            try:
                enc = tiktoken.encoding_for_model(embed_model)  # type: ignore[call-arg]
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")  # type: ignore[attr-defined]
            return len(enc.encode(s))
    except Exception:
        pass
    return max(1, len(s.split()))


def hard_wrap(text: str, *, window: int | None = None, overlap: int | None = None) -> list[str]:
    """Fallback char-based windowing for runaway chunks."""
    cfg = get_config()
    win = cfg.embed_hard_window if window is None else window
    ov = cfg.embed_hard_overlap if overlap is None else overlap
    if win <= 0:
        return [text]
    step = max(1, win - ov)
    out = []
    for i in range(0, len(text), step):
        out.append(text[i : i + win])
    return out


def enforce_chunk_size_limits(
    chunks: list[tuple[int, int, int, str]],
    *,
    embedding_provider: str | None,
    embed_model: str | None,
) -> list[tuple[int, int, int, str]]:
    """Ensure chunks are not excessively large by applying token-aware hard wrapping."""
    cfg = get_config()
    hard_chars = cfg.embed_hard_chars
    max_tokens = cfg.embeddings_max_tokens
    target_tokens = cfg.embeddings_target_tokens
    overlap_tokens = cfg.embeddings_token_overlap

    new_chunks: list[tuple[int, int, int, str]] = []
    next_idx = 0
    for _idx, start_line, end_line, chunk_text in chunks:
        if not chunk_text:
            new_chunks.append((next_idx, start_line, end_line, chunk_text))
            next_idx += 1
            continue

        toks = count_tokens(chunk_text, embed_model=embed_model, embedding_provider=embedding_provider)
        if toks > max_tokens:
            # Approximate char window size from tokens -> chars mapping
            try:
                chars_per_tok = max(1.0, len(chunk_text) / float(toks))
                window_chars = max(1, int(chars_per_tok * float(target_tokens)))
                overlap_chars = max(0, int(chars_per_tok * float(overlap_tokens)))
            except Exception:
                window_chars = cfg.embed_hard_window
                overlap_chars = cfg.embed_hard_overlap

            step = max(1, window_chars - overlap_chars)
            i = 0
            while i < len(chunk_text):
                win = chunk_text[i : i + window_chars]
                win_lines = win.count("\n") + 1
                new_chunks.append((next_idx, start_line, start_line + win_lines - 1, win))
                next_idx += 1
                start_line += win_lines
                i += step
        elif len(chunk_text) > hard_chars:
            windows = hard_wrap(chunk_text)
            for win in windows:
                win_lines = win.count("\n") + 1
                new_chunks.append((next_idx, start_line, start_line + win_lines - 1, win))
                next_idx += 1
                start_line += win_lines
        else:
            new_chunks.append((next_idx, start_line, end_line, chunk_text))
            next_idx += 1

    return new_chunks


def parse_jsonl_records(text: str, include_solution: bool | None = None) -> list[str]:
    """Parse a JSONL file into per-record strings suitable for embedding."""
    out: list[str] = []
    if not text:
        return out

    cfg = get_config()
    top_fields = [
        "task",
        "prompt",
        "statement",
        "description",
        "text",
        "content",
    ]
    sig_fields = ["signature", "sig", "signature_text"]
    docstring_fields = ["docstring", "doc", "explanation", "reasoning"]
    solution_fields = ["solution", "answer", "solution_text", "output"]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            if line:
                out.append(line)
            continue

        if not isinstance(obj, dict):
            out.append(str(obj))
            continue

        parts: list[str] = []

        for k in top_fields:
            if k in obj and obj.get(k):
                v = obj.get(k)
                if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
                    parts.append("\n".join(str(x) for x in v))
                else:
                    parts.append(str(v))
                break

        for k in sig_fields:
            if k in obj and obj.get(k):
                parts.append(str(obj.get(k)))
                break

        for k in docstring_fields:
            if k in obj and obj.get(k):
                parts.append(str(obj.get(k)))
                break

        sol = None
        for k in solution_fields:
            if k in obj and obj.get(k):
                sol = obj.get(k)
                break

        if not parts:
            for _, v in obj.items():
                if isinstance(v, str) and len(v) > 20:
                    parts.append(v)

        if parts:
            canonical = "\n\n".join(parts)
            if sol:
                use_solution = (
                    include_solution
                    if include_solution is not None
                    else cfg.embeddings_include_solution
                )
            else:
                use_solution = False

            if sol and use_solution:
                if isinstance(sol, Sequence) and not isinstance(sol, (str, bytes, bytearray)):
                    sol_text = "\n".join(str(x) for x in sol)
                else:
                    sol_text = str(sol)
                canonical = canonical + "\n\n--SOLUTION--\n\n" + sol_text

            out.append(canonical)
        else:
            try:
                compact = json.dumps(obj, separators=(",", ":"))
            except Exception:
                compact = str(obj)
            out.append(compact)

    return out


def chunk_text(
    text: str, max_lines: int = 100, overlap: int = 10
) -> list[tuple[int, int, int, str]]:
    """Split text into overlapping chunks with line tracking."""
    lines = text.splitlines()
    chunks = []
    i = 0
    chunk_idx = 0

    while i < len(lines):
        start = i
        end = min(i + max_lines, len(lines))
        if start >= end:
            break

        chunk_text = "\n".join(lines[start:end])
        chunks.append((chunk_idx, start + 1, end, chunk_text))  # 1-indexed lines
        chunk_idx += 1
        i += max_lines - overlap

    return chunks
