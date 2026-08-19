import os
import unittest
from unittest.mock import patch

from app.core.config import Settings


class SettingsTest(unittest.TestCase):
    def test_default_settings(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.app_name, "SmartGroceryAI")
        self.assertEqual(settings.app_env, "development")
        self.assertEqual(settings.log_level, "INFO")


if __name__ == "__main__":
    unittest.main()
