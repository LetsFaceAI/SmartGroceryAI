"""Unit tests for the grocery-assistant service without external model calls."""

from typing import cast
from unittest.mock import Mock

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from app.services.grocery_assistant import (
    EmptyModelResponseError,
    GroceryAssistantServiceError,
    get_grocery_assistant_response,
)


def make_model_mock(response: AIMessage) -> tuple[BaseChatModel, Mock]:
    """Create a typed model double that returns a predetermined local response."""
    model_mock = Mock(spec=BaseChatModel)
    model_mock.invoke.return_value = response
    return cast(BaseChatModel, model_mock), model_mock


def test_service_returns_text_and_sends_prompt_messages() -> None:
    """The service should send the prompt turn and expose trimmed assistant text."""
    model, model_mock = make_model_mock(
        AIMessage(content="  Compare the listed milk prices.  ")
    )

    result = get_grocery_assistant_response("Compare milk prices.", model=model)

    sent_messages = model_mock.invoke.call_args.args[0]
    assert len(sent_messages) == 2
    assert isinstance(sent_messages[0], SystemMessage)
    assert isinstance(sent_messages[1], HumanMessage)
    assert sent_messages[1].content == "Compare milk prices."
    assert result == "Compare the listed milk prices."


def test_service_rejects_empty_model_response() -> None:
    """Whitespace-only model output should not be treated as a valid answer."""
    model, _ = make_model_mock(AIMessage(content="   "))

    with pytest.raises(EmptyModelResponseError, match="empty response"):
        get_grocery_assistant_response("Find a milk deal.", model=model)


def test_service_wraps_model_invocation_failure() -> None:
    """Provider failures should retain their cause behind a stable service error."""
    model_mock = Mock(spec=BaseChatModel)
    provider_error = TimeoutError("provider timed out")
    model_mock.invoke.side_effect = provider_error

    with pytest.raises(
        GroceryAssistantServiceError,
        match="model invocation failed",
    ) as error_info:
        get_grocery_assistant_response(
            "Find a milk deal.",
            model=cast(BaseChatModel, model_mock),
        )

    assert error_info.value.__cause__ is provider_error
