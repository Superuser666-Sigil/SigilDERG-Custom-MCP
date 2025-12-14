# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Utilities to auto-start the Admin UI (Vite dev server) alongside the MCP server.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import get_config

logger = logging.getLogger("sigil_repos_mcp")


class AdminUIProcess:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.log_file: Optional[Path] = None

    def start(self):
        cfg = get_config()
        if not cfg.admin_enabled:
            logger.info("Admin UI autostart skipped: admin API disabled")
            return
        if not cfg.admin_ui_auto_start:
            logger.info("Admin UI autostart disabled by configuration")
            return

        ui_path = Path(cfg.admin_ui_path).expanduser().resolve()
        if not ui_path.exists():
            logger.warning("Admin UI path does not exist: %s", ui_path)
            return

        command = cfg.admin_ui_command or "npm"
        args = cfg.admin_ui_args or ["run", "dev"]
        port = cfg.admin_ui_port

        env = os.environ.copy()
        # Point UI at the admin backend by default
        env.setdefault("VITE_API_BASE_URL", f"http://{getattr(cfg, 'admin_host', cfg.server_host)}:{getattr(cfg, 'admin_port', cfg.server_port)}")
        env.setdefault("PORT", str(port))

        self.log_file = Path("/tmp/sigil_admin_ui.log")
        log_handle = self.log_file.open("w")

        try:
            self.proc = subprocess.Popen(
                [command, *args],
                cwd=str(ui_path),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            logger.info("Started Admin UI (PID %s) on port %s", self.proc.pid, port)
        except FileNotFoundError:
            logger.warning(
                "Failed to start Admin UI: command '%s' not found. "
                "Ensure Node/npm are installed and available.",
                command,
            )
            self.proc = None
            log_handle.close()
        except Exception as exc:
            logger.warning("Failed to start Admin UI: %s", exc)
            self.proc = None
            log_handle.close()

    def stop(self):
        if not self.proc:
            return
        try:
            self.proc.terminate()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


_ADMIN_UI_PROCESS = AdminUIProcess()


def start_admin_ui():
    try:
        _ADMIN_UI_PROCESS.start()
    except Exception as exc:
        logger.warning("Admin UI autostart failed: %s", exc)


def stop_admin_ui():
    try:
        _ADMIN_UI_PROCESS.stop()
    except Exception as exc:
        logger.warning("Failed to stop Admin UI cleanly: %s", exc)
