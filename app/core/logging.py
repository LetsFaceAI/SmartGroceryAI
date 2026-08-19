"""Standard-library logging setup shared by all application modules.

Centralizing the root logger's level and format gives every named logger consistent
output. The implementation intentionally uses only Python's ``logging`` package;
handlers and structured formatters can be added here later without changing calls
made by feature modules.
"""

import logging

from app.core.config import get_settings

# Include time, severity, and logger name so even basic local logs have context.
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Initialize process-wide logging.

    Args:
        level: Optional standard logging level. When omitted, the validated
            ``LOG_LEVEL`` application setting is used.

    ``basicConfig`` adds a default stream handler only when logging has not already
    been configured. This prevents repeated startup calls from duplicating output.
    """
    configured_level = level or get_settings().log_level
    logging.basicConfig(level=configured_level, format=LOG_FORMAT)

    # basicConfig does nothing when handlers already exist, so update the root level
    # explicitly to honor a new level on a repeated configuration call.
    logging.getLogger().setLevel(configured_level)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that inherits the centralized root configuration.

    Callers should normally pass ``__name__``. Hierarchical names identify the
    source module in output and allow targeted log-level overrides in the future.
    """
    return logging.getLogger(name)
