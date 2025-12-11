#!/usr/bin/env python3
"""Compatibility shim to rebuild indexes."""
from sigil_mcp.scripts.rebuild_indexes import main

if __name__ == "__main__":
    raise SystemExit(main())
