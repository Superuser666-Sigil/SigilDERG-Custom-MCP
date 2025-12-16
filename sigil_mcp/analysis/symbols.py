"""Symbol extraction utilities."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Symbol:
    """Represents a code symbol (function, class, variable, etc.)."""

    name: str
    kind: str
    file_path: str
    line: int
    signature: str | None = None
    scope: str | None = None


class SymbolExtractor:
    """Wrapper around universal-ctags for symbol extraction."""

    def __init__(self, *, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds

    def extract(self, file_path: Path, language: str) -> list[Symbol]:
        """Extract symbols from a file using universal-ctags."""
        try:
            result = subprocess.run(["ctags", "--version"], capture_output=True, timeout=1)
            if result.returncode != 0:
                return []
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("ctags not available, skipping symbol extraction")
            return []

        try:
            result = subprocess.run(
                [
                    "ctags",
                    "-f",
                    "-",
                    "--output-format=json",
                    "--fields=+n+S+s",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )

            if result.returncode != 0:
                return []

            symbols = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("_type") != "tag":
                    continue

                symbols.append(
                    Symbol(
                        name=data.get("name", ""),
                        kind=data.get("kind", "unknown"),
                        file_path=str(file_path),
                        line=data.get("line", 0),
                        signature=data.get("signature"),
                        scope=data.get("scope"),
                    )
                )
            return symbols
        except subprocess.TimeoutExpired:
            logger.warning("ctags timed out on %s", file_path)
            return []
        except Exception as exc:
            logger.debug("Error extracting symbols from %s: %s", file_path, exc)
            return []
