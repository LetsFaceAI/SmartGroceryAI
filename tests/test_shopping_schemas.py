"""Unit tests for grocery shopping schemas and their validation boundaries."""

import pytest
from pydantic import ValidationError

from app.schemas.shopping import ShoppingItem, ShoppingRequest


def test_shopping_request_accepts_and_normalizes_valid_input() -> None:
    """Valid grocery data should receive defaults and normalized whitespace."""
    # model_validate represents untyped data arriving from a future LLM or API and
    # demonstrates that nested item dictionaries become ShoppingItem instances.
    request = ShoppingRequest.model_validate(
        {
            "items": [
                {
                    "name": "  whole milk  ",
                    "quantity": 2,
                    "unit": " cartons ",
                    "notes": " lactose-free if available ",
                },
                {"name": "bananas"},
            ],
            "preferences": [" lowest price ", "organic when affordable"],
            "notes": " Shopping for one week ",
        }
    )

    assert request.items[0].name == "whole milk"
    assert request.items[0].quantity == 2
    assert request.items[0].unit == "cartons"
    assert request.items[1].quantity == 1.0
    assert request.items[1].unit is None
    assert request.preferences == ["lowest price", "organic when affordable"]
    assert request.notes == "Shopping for one week"


@pytest.mark.parametrize(
    ("item_data", "invalid_field"),
    [
        ({"name": "   "}, "name"),
        ({"name": "milk", "quantity": 0}, "quantity"),
        ({"name": "milk", "unit": "   "}, "unit"),
        ({"name": "milk", "unexpected": "value"}, "unexpected"),
    ],
)
def test_shopping_item_rejects_invalid_input(
    item_data: dict[str, object],
    invalid_field: str,
) -> None:
    """Invalid values and unknown keys should produce clear field errors."""
    with pytest.raises(ValidationError) as error_info:
        ShoppingItem.model_validate(item_data)

    error_locations = {str(error["loc"][0]) for error in error_info.value.errors()}
    assert invalid_field in error_locations


def test_shopping_request_requires_at_least_one_item() -> None:
    """An empty item collection cannot represent an actionable shopping request."""
    with pytest.raises(ValidationError) as error_info:
        ShoppingRequest(items=[])

    assert error_info.value.errors()[0]["loc"] == ("items",)


def test_shopping_request_does_not_share_preference_lists() -> None:
    """The default factory should give every request its own preference list."""
    first_request = ShoppingRequest.model_validate({"items": [{"name": "milk"}]})
    second_request = ShoppingRequest.model_validate({"items": [{"name": "bread"}]})

    first_request.preferences.append("organic")

    assert second_request.preferences == []
