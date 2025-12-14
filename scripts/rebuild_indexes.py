#!/usr/bin/env python3
#
# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""Compatibility shim to rebuild indexes."""
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-export helpers for callers that import from scripts.rebuild_indexes
from sigil_mcp.scripts.rebuild_indexes import (  # noqa: F401
    main,
    rebuild_all_indexes,
    rebuild_single_repo_index,
)

if __name__ == "__main__":
    raise SystemExit(main())
