"""Central logging configuration for the application."""

import logging

from app.core.config import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Configure root logging with an explicit level or the application setting."""
    configured_level = level or get_settings().log_level
    logging.basicConfig(level=configured_level, format=LOG_FORMAT)

    # basicConfig is intentionally idempotent, but the level may change between calls.
    logging.getLogger().setLevel(configured_level)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for an application module."""
    return logging.getLogger(name)
