#!/usr/bin/env python3
"""Simple harness to exercise the server offload path for manual/integration testing.

This script replaces the real worker function with a lightweight stub so it can be
run reliably in CI or developer machines without requiring full Sigil initialization.

Usage:
    python tests/harness/run_offload_harness.py

The script will create a temporary repo, submit several file-change events and
verify the offload executor invokes the worker stub for each file.
"""
from __future__ import annotations

import concurrent.futures
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import sigil_mcp.server as server


def main() -> int:
    with TemporaryDirectory() as td:
        td = Path(td)
        repo_name = "harness_repo"
        repo_root = td / repo_name
        repo_root.mkdir()
        server.REPOS[repo_name] = repo_root

        # Replace process_pool with a ThreadPoolExecutor for harness determinism
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        server.process_pool = ex
        server.USE_PROCESS_POOL = True

        called = []

        def fake_task(cfg_json, rn, rr, fp):
            called.append((rn, rr, fp))
            # simulate some work
            time.sleep(0.1)
            return True

        server.index_file_task = fake_task

        # Create multiple files and fire change events
        files = []
        for i in range(8):
            p = repo_root / f"file_{i}.py"
            p.write_text(f"print({i})\n")
            files.append(p)

        for p in files:
            server._on_file_change(repo_name, p, "modified")

        # Wait for tasks to complete
        ex.shutdown(wait=True)

        print(f"Tasks invoked: {len(called)} expected {len(files)}")
        if len(called) != len(files):
            print("Harness failed: not all tasks were executed")
            return 2

        print("Harness succeeded")
        return 0


if __name__ == "__main__":
    sys.exit(main())
