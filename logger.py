"""
Mimi — Shared logging configuration.

Every module in the Mimi pipeline imports get_logger() from here to ensure
consistent log format, level, and structured output across all pipeline stages.

Usage:
    from mimi.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Stage complete", extra={"stage": "asr", "duration_s": 4.2})
"""

import logging
import os
import sys
from typing import Optional


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger() -> None:
    """Configure the root logger once. Subsequent calls are no-ops."""
    global _configured
    if _configured:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger with Mimi's standard formatting applied.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    _configure_root_logger()
    return logging.getLogger(name or "mimi")
