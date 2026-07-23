"""Centralized logging configuration with intercept-style error capture."""

from __future__ import annotations

import logging
import sys
from typing import Final

_CONFIGURED: Final[bool] = False


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger once and return the application logger."""
    global _CONFIGURED  # noqa: PLW0603

    app_logger = logging.getLogger("agrivoltaics")
    if _CONFIGURED:
        return app_logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()
    root.addHandler(handler)

    app_logger.setLevel(numeric_level)
    _CONFIGURED = True
    return app_logger
