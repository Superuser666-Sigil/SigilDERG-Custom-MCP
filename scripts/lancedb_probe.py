"""Diagnostic probe for LanceDB connect hangs.

Runs `lancedb.connect` with verbose logging and a timeout, printing any output
seen on stdout/stderr. Use this to capture clues when connect blocks silently.
"""

from __future__ import annotations

import os
import sys
import time
import traceback


def main() -> None:
    os.environ.setdefault("LANCEDB_LOG", "trace")
    os.environ.setdefault("LANCEDB_LOG_STYLE", "never")
    os.environ.setdefault("LANCE_DISABLE_IO_URING", "1")

    import lancedb  # noqa: WPS433

    target = os.environ.get("LANCEDB_PROBE_PATH", "tmp_lancedb_probe")
    timeout_seconds = float(os.environ.get("LANCEDB_PROBE_TIMEOUT", "15"))

    print(f"Starting probe: path={target}, timeout={timeout_seconds}s")
    sys.stdout.flush()

    start = time.monotonic()
    try:
        db = lancedb.connect(target)
        elapsed = time.monotonic() - start
        print(f"SUCCESS: connect returned in {elapsed:.2f}s -> {db}")
    except Exception:
        elapsed = time.monotonic() - start
        print(f"ERROR after {elapsed:.2f}s:")
        traceback.print_exc()
    finally:
        sys.stdout.flush()

    hung = (time.monotonic() - start) > timeout_seconds
    if hung:
        print(f"TIMEOUT: connect exceeded {timeout_seconds}s without returning.")
        sys.exit(2)


if __name__ == "__main__":
    main()
