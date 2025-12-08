# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

#!/usr/bin/env python3
"""
Entry point for the Sigil MCP Admin API service.

Starts a separate HTTP server for operational management endpoints.
"""

import logging
import sys
from pathlib import Path

import uvicorn

# Always use absolute imports - works for both module and script execution
# Add parent directory to path if needed
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from sigil_mcp.admin_api import app
from sigil_mcp.config import get_config
from sigil_mcp.logging_setup import setup_logging, get_log_file_path

# Set up logging (will use same config as main server)
config = get_config()
log_file_path = get_log_file_path(config.log_file)
setup_logging(
    log_file=str(log_file_path) if log_file_path else None,
    log_level=config.log_level,
    console_output=True,
)

logger = logging.getLogger("sigil_admin_main")


def main():
    """Start the Admin API server."""
    config = get_config()
    
    if not config.admin_enabled:
        logger.error("Admin API is disabled in configuration")
        logger.error("Set admin.enabled=true in config.json or SIGIL_MCP_ADMIN_ENABLED=true")
        sys.exit(1)
    
    host = config.admin_host
    port = config.admin_port
    
    logger.info("=" * 60)
    logger.info("Sigil MCP Admin API")
    logger.info("=" * 60)
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  GET  /                    - API information")
    logger.info("  GET  /admin/status        - Server status")
    logger.info("  GET  /admin/index/stats   - Index statistics")
    logger.info("  POST /admin/index/rebuild - Rebuild index")
    logger.info("  POST /admin/vector/rebuild - Rebuild vector index")
    logger.info("  GET  /admin/logs/tail     - View logs")
    logger.info("  GET  /admin/config        - View configuration")
    logger.info("")
    
    if config.admin_api_key:
        logger.info(f"API Key authentication: ENABLED")
        logger.info(f"Use header: X-Admin-Key: <your-key>")
    else:
        logger.info("API Key authentication: DISABLED")
    
    allowed_ips = config.admin_allowed_ips
    if allowed_ips:
        logger.info(f"IP Whitelist: {', '.join(allowed_ips)}")
    logger.info("")
    logger.info("Starting server...")
    logger.info("=" * 60)
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Admin API...")
    except Exception as e:
        logger.error(f"Failed to start Admin API: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
