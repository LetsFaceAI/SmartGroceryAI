"""Unit tests for centralized chat-model configuration."""

import pytest

from app.core.config import Settings
from app.core.llm import LLMConfigurationError, create_chat_model


def test_create_chat_model_requires_api_key() -> None:
    """Model construction should explain missing credentials before network access."""
    settings = Settings(openai_api_key=None)

    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        create_chat_model(settings)
