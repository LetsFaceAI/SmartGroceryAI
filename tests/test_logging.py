"""Tests for the centralized standard-library logging setup."""

import logging
import unittest

from app.core.logging import configure_logging, get_logger


class LoggingTest(unittest.TestCase):
    """Verify logging setup without leaking global state into other tests."""

    def test_logging_configuration_initializes(self) -> None:
        """Configuration should initialize and create usable named loggers."""
        root_logger = logging.getLogger()

        # Logging is global process state. Preserve it so this test remains isolated
        # from unittest and any future test runner configuration.
        original_level = root_logger.level
        original_handlers = root_logger.handlers.copy()

        try:
            root_logger.handlers.clear()

            configure_logging(level="DEBUG")

            self.assertEqual(root_logger.level, logging.DEBUG)
            self.assertIsInstance(get_logger("smartgroceryai.test"), logging.Logger)
        finally:
            # Restore handlers even when an assertion fails.
            root_logger.handlers.clear()
            root_logger.handlers.extend(original_handlers)
            root_logger.setLevel(original_level)


if __name__ == "__main__":
    unittest.main()
