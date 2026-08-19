"""Unit tests for chat-model configuration and message invocation boundaries."""

from typing import cast
from unittest.mock import Mock

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings
from app.core.llm import LLMConfigurationError, create_chat_model, send_message


def test_create_chat_model_requires_api_key() -> None:
    """Model construction should explain missing credentials before network access."""
    settings = Settings(openai_api_key=None)

    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        create_chat_model(settings)


def test_send_message_uses_grocery_prompt_messages() -> None:
    """The helper should send system and human messages and return an AIMessage."""
    expected_response = AIMessage(content="Milk is on the shopping list.")
    model_mock = Mock(spec=BaseChatModel)
    model_mock.invoke.return_value = expected_response

    response = send_message(
        "Add milk to my shopping list.",
        model=cast(BaseChatModel, model_mock),
    )

    sent_messages = model_mock.invoke.call_args.args[0]
    assert len(sent_messages) == 2
    assert isinstance(sent_messages[0], SystemMessage)
    assert isinstance(sent_messages[1], HumanMessage)
    assert sent_messages[1].content == "Add milk to my shopping list."
    assert response is expected_response
