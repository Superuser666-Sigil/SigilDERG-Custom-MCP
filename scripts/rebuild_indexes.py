# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

#!/usr/bin/env python3
"""Compatibility shim to rebuild indexes."""
from sigil_mcp.scripts.rebuild_indexes import main

if __name__ == "__main__":
    raise SystemExit(main())
