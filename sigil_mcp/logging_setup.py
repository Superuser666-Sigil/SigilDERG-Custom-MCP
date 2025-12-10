# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Centralized logging configuration for Sigil MCP Server.

Configures file and console logging for all modules in the application.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    console_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Configure logging for all Sigil MCP Server modules.
    
    Sets up:
    - File logging (if log_file is provided) with rotation
    - Console logging (stdout/stderr) if console_output is True
    - Consistent formatting across all loggers
    - Proper log levels for all modules
    
    Args:
        log_file: Path to log file. If None, no file logging.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to also log to console
        max_bytes: Maximum log file size before rotation (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add file handler if log_file is specified
    if log_file:
        log_path = Path(log_file).expanduser()
        log_dir = log_path.parent
        
        # Create log directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use RotatingFileHandler to prevent log files from growing too large
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logging.getLogger(__name__).info(f"File logging enabled: {log_path}")
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Configure specific loggers with appropriate levels
    # Main server logger
    logging.getLogger("sigil_repos_mcp").setLevel(numeric_level)
    
    # Admin API loggers
    logging.getLogger("sigil_admin").setLevel(numeric_level)
    logging.getLogger("sigil_admin_main").setLevel(numeric_level)
    
    # Module loggers (will inherit from root)
    for module_name in [
        "sigil_mcp.config",
        "sigil_mcp.indexer",
        "sigil_mcp.oauth",
        "sigil_mcp.auth",
        "sigil_mcp.watcher",
        "sigil_mcp.embeddings",
        "sigil_mcp.llamacpp_provider",
    ]:
        logging.getLogger(module_name).setLevel(numeric_level)
    
    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


def get_log_file_path(config_log_file: Optional[str] = None) -> Optional[Path]:
    """
    Get the log file path, creating default if needed.
    
    Args:
        config_log_file: Log file path from config (can be None)
        
    Returns:
        Path to log file, or None if no file logging should be used
    """
    if config_log_file:
        return Path(config_log_file).expanduser()
    
    # Default location
    default_path = Path.home() / ".sigil_mcp_server" / "logs" / "server.log"
    return default_path

