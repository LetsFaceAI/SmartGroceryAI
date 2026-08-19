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


@pytest.mark.parametrize(
    ("user_input", "structured_payload"),
    [
        pytest.param(
            "Milk.",
            {"items": [{"name": "milk"}]},
            id="single-item",
        ),
        pytest.param(
            "I need milk, eggs, and blueberries.",
            {
                "items": [
                    {"name": "milk"},
                    {"name": "eggs"},
                    {"name": "blueberries"},
                ]
            },
            id="multiple-items",
        ),
        pytest.param(
            "I need 3 apples.",
            {"items": [{"name": "apples", "quantity": 3}]},
            id="quantity",
        ),
        pytest.param(
            "Add 1.5 kg of potatoes.",
            {"items": [{"name": "potatoes", "quantity": 1.5, "unit": "kg"}]},
            id="unit",
        ),
        pytest.param(
            "I prefer organic groceries and need 2% milk.",
            {
                "items": [{"name": "milk", "notes": "2%"}],
                "preferences": ["organic"],
            },
            id="preferences-and-item-qualifier",
        ),
        pytest.param(
            "I need some fruit.",
            {"items": [{"name": "fruit"}]},
            id="vague-partial-input",
        ),
    ],
)
def test_extract_shopping_request_handles_focused_inputs(
    user_input: str,
    structured_payload: dict[str, object],
) -> None:
    """Representative inputs should always cross the service as validated schemas."""
    expected_request = ShoppingRequest.model_validate(structured_payload)
    model, model_mock, structured_model_mock = make_structured_model_mock(
        # Returning a raw payload verifies that the service, not only the mocked LLM,
        # enforces the Pydantic schema before exposing data to callers.
        result=structured_payload
    )

    result = extract_shopping_request(user_input, model=model)

    model_mock.with_structured_output.assert_called_once_with(ShoppingRequest)
    sent_messages = structured_model_mock.invoke.call_args.args[0]
    assert len(sent_messages) == 2
    assert isinstance(sent_messages[0], SystemMessage)
    assert isinstance(sent_messages[1], HumanMessage)
    assert sent_messages[1].content == user_input
    assert isinstance(result, ShoppingRequest)
    assert result == expected_request


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
