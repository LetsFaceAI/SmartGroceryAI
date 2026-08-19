"""Tests for loading and validating application configuration."""

import os
import unittest
from unittest.mock import patch

from app.core.config import Settings


class SettingsTest(unittest.TestCase):
    """Verify the baseline configuration independently of the developer machine."""

    def test_default_settings(self) -> None:
        """Settings should use documented defaults when no external values exist."""
        # Clear the process environment and disable .env loading so a developer's
        # local configuration cannot make this default-value test nondeterministic.
        with patch.dict(os.environ, {}, clear=True):
            # Pydantic Settings supports _env_file at runtime, but its generated
            # constructor signature does not expose this test-only option to mypy.
            settings = Settings(_env_file=None)  # type: ignore[call-arg]

        self.assertEqual(settings.app_name, "SmartGroceryAI")
        self.assertEqual(settings.app_env, "development")
        self.assertEqual(settings.log_level, "INFO")


if __name__ == "__main__":
    unittest.main()
