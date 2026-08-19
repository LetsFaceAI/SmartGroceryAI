import logging
import unittest

from app.core.logging import configure_logging, get_logger


class LoggingTest(unittest.TestCase):
    def test_logging_configuration_initializes(self) -> None:
        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_handlers = root_logger.handlers.copy()

        try:
            root_logger.handlers.clear()

            configure_logging(level="DEBUG")

            self.assertEqual(root_logger.level, logging.DEBUG)
            self.assertIsInstance(get_logger("smartgroceryai.test"), logging.Logger)
        finally:
            root_logger.handlers.clear()
            root_logger.handlers.extend(original_handlers)
            root_logger.setLevel(original_level)


if __name__ == "__main__":
    unittest.main()
