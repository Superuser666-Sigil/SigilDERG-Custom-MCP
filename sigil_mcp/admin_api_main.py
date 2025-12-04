import logging
import uvicorn
from .config import get_config
from .admin_api import app as admin_app

logger = logging.getLogger("sigil_admin_main")

def main() -> None:
    cfg = get_config()

    if not cfg.admin_enabled:
        logger.warning("Admin API is disabled in config (admin.enabled=false)")
        return

    host = cfg.admin_host
    port = cfg.admin_port

    logger.info("Starting Sigil MCP Admin API on %s:%s", host, port)
    logger.info("This API is intended for localhost-only access.")

    uvicorn.run(
        admin_app,
        host=host,
        port=port,
        log_level=cfg.log_level.lower(),
    )


if __name__ == "__main__":
    main()
