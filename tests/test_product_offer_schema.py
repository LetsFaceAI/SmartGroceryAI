"""Unit tests for normalized grocery flyer product offers."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.product_offer import (
    MeasurementUnit,
    ProductOffer,
    PromotionStatus,
)


def test_product_offer_accepts_complete_valid_data() -> None:
    """A complete flyer record should normalize strings, money, and dates."""
    # model_validate mirrors the raw dictionary that a future provider adapter will
    # create before the rest of the application receives the offer.
    offer = ProductOffer.model_validate(
        {
            "product_name": " 2% Milk ",
            "brand": " Dairy Best ",
            "store": " Downtown Grocer ",
            "price": "4.99",
            "regular_price": "5.79",
            "currency": "CAD",
            "package_size": "4",
            "unit": " L ",
            "promotion_status": "sale",
            "valid_from": "2026-08-20",
            "valid_until": "2026-08-26",
            "source": "flipp-flyer-record-123",
        }
    )

    assert offer.product_name == "2% Milk"
    assert offer.brand == "Dairy Best"
    assert offer.store == "Downtown Grocer"
    assert offer.price == Decimal("4.99")
    assert offer.regular_price == Decimal("5.79")
    assert offer.package_size == Decimal("4")
    assert offer.unit is MeasurementUnit.LITRE
    assert offer.promotion_status is PromotionStatus.SALE
    assert offer.valid_from == date(2026, 8, 20)
    assert offer.valid_until == date(2026, 8, 26)


def test_product_offer_accepts_partial_flyer_data() -> None:
    """Missing optional metadata should remain explicit rather than being invented."""
    offer = ProductOffer(
        product_name="Blueberries",
        store="Neighbourhood Market",
        price=Decimal("2.50"),
        source="weekly-flyer",
    )

    assert offer.brand is None
    assert offer.regular_price is None
    assert offer.currency == "CAD"
    assert offer.package_size is None
    assert offer.unit is None
    assert offer.promotion_status is PromotionStatus.UNKNOWN
    assert offer.valid_from is None
    assert offer.valid_until is None


@pytest.mark.parametrize(
    ("offer_data", "invalid_field"),
    [
        (
            {"product_name": " ", "store": "Market", "price": "1.99", "source": "x"},
            "product_name",
        ),
        (
            {"product_name": "Milk", "store": " ", "price": "1.99", "source": "x"},
            "store",
        ),
        (
            {
                "product_name": "Milk",
                "store": "Market",
                "price": "0.00",
                "source": "x",
            },
            "price",
        ),
        (
            {
                "product_name": "Milk",
                "store": "Market",
                "price": "-0.01",
                "source": "x",
            },
            "price",
        ),
        (
            {
                "product_name": "Milk",
                "store": "Market",
                "price": "1.999",
                "source": "x",
            },
            "price",
        ),
        (
            {
                "product_name": "Milk",
                "store": "Market",
                "price": "1.99",
                "package_size": 0,
                "source": "x",
            },
            "package_size",
        ),
        (
            {
                "product_name": "Milk",
                "store": "Market",
                "price": "1.99",
                "currency": "cad",
                "source": "x",
            },
            "currency",
        ),
        (
            {
                "product_name": "Milk",
                "store": "Market",
                "price": "1.99",
                "source": " ",
            },
            "source",
        ),
        (
            {
                "product_name": "Milk",
                "store": "Market",
                "price": "1.99",
                "source": "x",
                "unexpected": "value",
            },
            "unexpected",
        ),
    ],
)
def test_product_offer_rejects_invalid_fields(
    offer_data: dict[str, object],
    invalid_field: str,
) -> None:
    """Invalid field values should fail before an offer enters application logic."""
    with pytest.raises(ValidationError) as error_info:
        ProductOffer.model_validate(offer_data)

    error_locations = {str(error["loc"][0]) for error in error_info.value.errors()}
    assert invalid_field in error_locations


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("promotion_status", "discounted"),
        ("unit", "litres"),
    ],
)
def test_product_offer_rejects_unknown_enum_values(
    field_name: str,
    invalid_value: str,
) -> None:
    """Normalized status and unit fields should accept only supported enum values."""
    offer_data: dict[str, object] = {
        "product_name": "Milk",
        "store": "Market",
        "price": "3.99",
        "source": "weekly-flyer",
        field_name: invalid_value,
    }

    with pytest.raises(ValidationError) as error_info:
        ProductOffer.model_validate(offer_data)

    assert error_info.value.errors()[0]["loc"] == (field_name,)


def test_product_offer_rejects_reversed_validity_window() -> None:
    """An offer cannot expire before its advertised start date."""
    with pytest.raises(ValidationError, match="valid_until"):
        ProductOffer.model_validate(
            {
                "product_name": "Eggs",
                "store": "Market",
                "price": "3.49",
                "valid_from": "2026-08-27",
                "valid_until": "2026-08-20",
                "source": "weekly-flyer",
            }
        )


@pytest.mark.parametrize("sale_price", ["3.99", "4.99"])
def test_product_offer_rejects_invalid_sale_price_relationship(
    sale_price: str,
) -> None:
    """Sale pricing must be strictly lower than a supplied regular price."""
    with pytest.raises(ValidationError, match="sale price"):
        ProductOffer.model_validate(
            {
                "product_name": "Eggs",
                "store": "Market",
                "price": sale_price,
                "regular_price": "3.99",
                "promotion_status": "sale",
                "source": "weekly-flyer",
            }
        )


def test_product_offer_rejects_mismatched_regular_status_price() -> None:
    """Regular status should not accompany two different current price values."""
    with pytest.raises(ValidationError, match="Regular-status"):
        ProductOffer.model_validate(
            {
                "product_name": "Eggs",
                "store": "Market",
                "price": "3.49",
                "regular_price": "3.99",
                "promotion_status": "regular",
                "source": "weekly-flyer",
            }
        )


def test_product_offer_rejects_unknown_status_price_above_regular_price() -> None:
    """Unknown promotion data must not permit contradictory price values."""
    with pytest.raises(ValidationError, match="Unknown-status price"):
        ProductOffer.model_validate(
            {
                "product_name": "Eggs",
                "store": "Market",
                "price": "4.99",
                "regular_price": "3.99",
                # Omitting promotion_status exercises its UNKNOWN default.
                "source": "weekly-flyer",
            }
        )
