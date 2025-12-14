# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Sigil MCP Server - Model Context Protocol server for code navigation.

Provides IDE-like code search and navigation capabilities via MCP, including:
- Fast trigram-based text search
- Symbol-based navigation (functions, classes, methods)
- Semantic code search using vector embeddings
- OAuth 2.0 authentication for remote access
"""

__version__ = "1.0.1"
__author__ = "Sigil DERG"

from sigil_mcp.config import get_config
from sigil_mcp.indexer import SearchResult, SigilIndex, Symbol

__all__ = [
    "SigilIndex",
    "Symbol",
    "SearchResult",
    "get_config",
    "__version__",
]
