"""Unit tests for mapping external flyer dictionaries into product offers."""

from decimal import Decimal

import pytest

from app.schemas.product_offer import (
    MeasurementUnit,
    ProductOffer,
    PromotionStatus,
)
from app.services.product_offer_mapper import (
    ProductOfferMappingError,
    map_product_offer,
)


def test_map_product_offer_maps_complete_external_data() -> None:
    """Known external keys and aliases should become one validated offer."""
    raw_offer: dict[str, object] = {
        "product": " 2% Milk ",
        "brand": " Dairy Best ",
        "store": " Downtown Grocer ",
        "price": "4.99",
        "regularPrice": "5.79",
        "currency": "CAD",
        "packageSize": "4",
        "unit": " litres ",
        "onSale": True,
        "validFrom": "2026-08-20",
        "validUntil": "2026-08-26",
        "source": "external-record-123",
        # Unused provider metadata does not leak into the application schema.
        "providerInternalId": "internal-456",
    }

    offer = map_product_offer(raw_offer)

    assert isinstance(offer, ProductOffer)
    assert offer.product_name == "2% Milk"
    assert offer.brand == "Dairy Best"
    assert offer.store == "Downtown Grocer"
    assert offer.price == Decimal("4.99")
    assert offer.regular_price == Decimal("5.79")
    assert offer.package_size == Decimal("4")
    assert offer.unit is MeasurementUnit.LITRE
    assert offer.promotion_status is PromotionStatus.SALE
    assert offer.source == "external-record-123"


def test_map_product_offer_handles_missing_optional_data() -> None:
    """Minimal valid data should use ProductOffer's documented defaults."""
    offer = map_product_offer(
        {
            "product": "Blueberries",
            "store": "Neighbourhood Market",
            "price": "2.50",
            "source": "weekly-flyer",
        }
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
    ("raw_offer", "expected_error_field"),
    [
        (
            {"product": "Milk", "price": "3.99", "source": "flyer"},
            "store",
        ),
        (
            {
                "product": "Milk",
                "store": "Market",
                "price": "0.00",
                "source": "flyer",
            },
            "price",
        ),
        (
            {
                "product": "Milk",
                "store": "Market",
                "price": "3.99",
                "validFrom": "2026-08-27",
                "validUntil": "2026-08-20",
                "source": "flyer",
            },
            "offer",
        ),
    ],
)
def test_map_product_offer_wraps_schema_validation_errors(
    raw_offer: dict[str, object],
    expected_error_field: str,
) -> None:
    """Invalid required and cross-field data should raise one mapping error type."""
    with pytest.raises(
        ProductOfferMappingError,
        match=expected_error_field,
    ) as error_info:
        map_product_offer(raw_offer)

    assert error_info.value.__cause__ is not None


def test_map_product_offer_rejects_non_boolean_sale_flag() -> None:
    """Ambiguous external sale flags should fail instead of being guessed."""
    with pytest.raises(ProductOfferMappingError, match="onSale"):
        map_product_offer(
            {
                "product": "Milk",
                "store": "Market",
                "price": "3.99",
                "onSale": "yes",
                "source": "flyer",
            }
        )
