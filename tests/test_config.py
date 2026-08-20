"""Tests for loading and validating application configuration."""

import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

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
        self.assertIsNone(settings.openai_api_key)
        self.assertEqual(settings.openai_model, "gpt-5-nano")
        self.assertIsNone(settings.apify_mcp_server_url)
        self.assertIsNone(settings.apify_api_token)
        self.assertEqual(settings.apify_mcp_tool_timeout_seconds, 120.0)
        self.assertEqual(settings.search_max_items_per_request, 20)
        self.assertEqual(settings.search_max_external_actor_calls_per_request, 3)
        self.assertEqual(settings.search_max_concurrency, 1)

    def test_search_budget_settings_enforce_absolute_caps(self) -> None:
        """Environment configuration cannot disable application safety ceilings."""
        invalid_overrides = (
            {"search_max_items_per_request": 101},
            {"search_max_external_actor_calls_per_request": 11},
            {"search_max_concurrency": 6},
        )

        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
