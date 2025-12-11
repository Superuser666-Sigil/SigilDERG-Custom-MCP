# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Optional auto-install support for external MCP servers.

Intended for convenience in dev: when enabled, will run npx/npm/bunx commands
declared in external_mcp_servers with auto_install=true or command in a known set.
"""

from __future__ import annotations

import subprocess
import logging
from typing import Iterable, Dict, Any

logger = logging.getLogger("sigil_repos_mcp")

KNOWN_INSTALL_COMMANDS = {"npx", "npm", "bunx"}


def auto_install(servers: Iterable[Dict[str, Any]]) -> None:
    """
    Run installation commands for eligible servers. Safe no-op on errors.
    """
    for server in servers:
        if server.get("disabled"):
            continue
        command = server.get("command")
        args = server.get("args", [])
        auto_flag = bool(server.get("auto_install", False))
        if not command:
            continue
        if command not in KNOWN_INSTALL_COMMANDS and not auto_flag:
            continue

        cmdline = [command, *args]
        try:
            logger.info("Auto-installing MCP server '%s' via: %s", server.get("name"), " ".join(cmdline))
            subprocess.run(cmdline, check=True)
        except FileNotFoundError:
            logger.warning("Install command not found for server '%s': %s", server.get("name"), command)
        except subprocess.CalledProcessError as exc:
            logger.warning("Install command failed for server '%s': %s", server.get("name"), exc)
        except Exception as exc:
            logger.warning("Unexpected error auto-installing server '%s': %s", server.get("name"), exc)
