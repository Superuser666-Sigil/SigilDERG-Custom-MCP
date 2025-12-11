# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import logging
from pathlib import Path

from sigil_mcp.logging_setup import get_log_file_path, setup_logging


def test_get_log_file_path_defaults(monkeypatch):
    monkeypatch.delenv("SIGIL_MCP_LOG_FILE", raising=False)
    default = get_log_file_path(None)
    assert default.name == "server.log"

    custom = get_log_file_path("~/tmp/logs/custom.log")
    assert custom == Path.home() / "tmp/logs/custom.log"


def test_setup_logging_writes_file(monkeypatch, tmp_path):
    log_path = tmp_path / "logdir" / "server.log"
    setup_logging(log_file=str(log_path), log_level="INFO", console_output=False)

    logger = logging.getLogger("sigil_repos_mcp")
    logger.info("hello-file")

    # Flush handlers
    for handler in logger.handlers:
        handler.flush()

    assert log_path.exists()
    contents = log_path.read_text()
    assert "hello-file" in contents
