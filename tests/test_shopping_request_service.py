"""Offline unit tests for structured grocery request extraction."""

from typing import cast
from unittest.mock import Mock

import pytest
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from app.schemas.shopping import ShoppingRequest
from app.services.shopping_request import (
    ShoppingRequestExtractionError,
    ShoppingRequestValidationError,
    extract_shopping_request,
)


def make_structured_model_mock(
    *,
    result: object | None = None,
    error: Exception | None = None,
) -> tuple[BaseChatModel, Mock, Mock]:
    """Create base and structured model doubles without provider initialization."""
    model_mock = Mock(spec=BaseChatModel)
    structured_model_mock = Mock()
    model_mock.with_structured_output.return_value = structured_model_mock

    if error is not None:
        structured_model_mock.invoke.side_effect = error
    else:
        structured_model_mock.invoke.return_value = result

    return cast(BaseChatModel, model_mock), model_mock, structured_model_mock


def test_extract_shopping_request_returns_validated_schema() -> None:
    """Natural-language input should flow through the bound structured model."""
    expected_request = ShoppingRequest.model_validate(
        {
            "items": [
                {"name": "milk", "quantity": 2, "unit": "bags"},
                {"name": "eggs"},
                {"name": "blueberries"},
            ]
        }
    )
    model, model_mock, structured_model_mock = make_structured_model_mock(
        result=expected_request
    )

    result = extract_shopping_request(
        "I need 2 bags of milk, eggs, and blueberries.",
        model=model,
    )

    model_mock.with_structured_output.assert_called_once_with(ShoppingRequest)
    sent_messages = structured_model_mock.invoke.call_args.args[0]
    assert len(sent_messages) == 2
    assert isinstance(sent_messages[0], SystemMessage)
    assert isinstance(sent_messages[1], HumanMessage)
    assert sent_messages[1].content == ("I need 2 bags of milk, eggs, and blueberries.")
    assert result is expected_request


def test_extract_shopping_request_rejects_invalid_structured_data() -> None:
    """A payload without any items should fail the ShoppingRequest contract."""
    model, _, _ = make_structured_model_mock(result={"items": []})

    with pytest.raises(
        ShoppingRequestValidationError,
        match="did not match",
    ) as error_info:
        extract_shopping_request("I need groceries.", model=model)

    assert error_info.value.__cause__ is not None


def test_extract_shopping_request_wraps_model_failure() -> None:
    """Provider errors should remain available behind a stable extraction error."""
    provider_error = TimeoutError("provider timed out")
    model, _, _ = make_structured_model_mock(error=provider_error)

    with pytest.raises(
        ShoppingRequestExtractionError,
        match="model invocation failed",
    ) as error_info:
        extract_shopping_request("I need milk.", model=model)

    assert error_info.value.__cause__ is provider_error
