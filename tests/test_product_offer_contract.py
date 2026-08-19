"""Focused contract tests for the Phase 4 external-offer data boundary.

These tests intentionally exercise both direct Pydantic validation and mapper output.
They document the guarantees downstream comparison and AI code can rely on before an
MCP provider is introduced.
"""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.product_offer import (
    MeasurementUnit,
    ProductOffer,
    PromotionStatus,
)
from app.services.product_offer_mapper import (
    ProductOfferMappingError,
    map_product_offer,
)


def test_contract_accepts_complete_offer() -> None:
    """Every supported field should produce one precise, typed offer."""
    offer = ProductOffer.model_validate(
        {
            "product_name": "2% Milk",
            "brand": "Dairy Best",
            "store": "Downtown Grocer",
            "price": "4.99",
            "regular_price": "5.79",
            "currency": "CAD",
            "package_size": "4",
            "unit": "L",
            "promotion_status": "sale",
            "valid_from": "2026-08-20",
            "valid_until": "2026-08-26",
            "source": "flyer-record-123",
        }
    )

    assert offer.price == Decimal("4.99")
    assert offer.unit is MeasurementUnit.LITRE
    assert offer.promotion_status is PromotionStatus.SALE


def test_contract_accepts_partial_offer() -> None:
    """Only identity, price, and provenance are required at this boundary."""
    offer = ProductOffer.model_validate(
        {
            "product_name": "Blueberries",
            "store": "Neighbourhood Market",
            "price": "2.50",
            "source": "weekly-flyer",
        }
    )

    assert offer.brand is None
    assert offer.regular_price is None
    assert offer.unit is None
    assert offer.promotion_status is PromotionStatus.UNKNOWN
    assert offer.valid_from is None
    assert offer.valid_until is None


@pytest.mark.parametrize("invalid_price", ["0.00", "-1.00", "1.999"])
def test_contract_rejects_invalid_prices(invalid_price: str) -> None:
    """Zero, negative, and over-precise prices must not cross the boundary."""
    with pytest.raises(ValidationError) as error_info:
        ProductOffer.model_validate(
            {
                "product_name": "Milk",
                "store": "Market",
                "price": invalid_price,
                "source": "weekly-flyer",
            }
        )

    assert error_info.value.errors()[0]["loc"] == ("price",)


def test_contract_rejects_invalid_date_range() -> None:
    """The final validity date cannot be earlier than the starting date."""
    with pytest.raises(ValidationError, match="valid_until"):
        ProductOffer.model_validate(
            {
                "product_name": "Milk",
                "store": "Market",
                "price": "3.99",
                "valid_from": "2026-08-27",
                "valid_until": "2026-08-20",
                "source": "weekly-flyer",
            }
        )


def test_mapper_output_exactly_matches_product_offer_contract() -> None:
    """Mapper callers should receive the same model as direct schema validation."""
    mapped_offer = map_product_offer(
        {
            "product": "2% Milk",
            "store": "Downtown Grocer",
            "price": "4.99",
            "regularPrice": "5.79",
            "packageSize": "4",
            "unit": "L",
            "onSale": True,
            "source": "flyer-record-123",
        }
    )
    expected_offer = ProductOffer.model_validate(
        {
            "product_name": "2% Milk",
            "store": "Downtown Grocer",
            "price": "4.99",
            "regular_price": "5.79",
            "package_size": "4",
            "unit": "L",
            "promotion_status": "sale",
            "source": "flyer-record-123",
        }
    )

    assert type(mapped_offer) is ProductOffer
    assert mapped_offer == expected_offer


@pytest.mark.parametrize("malformed_offer", [None, [], "not an offer", 42])
def test_contract_rejects_non_mapping_external_data(malformed_offer: object) -> None:
    """Malformed top-level provider values should fail with a stable mapper error."""
    with pytest.raises(ProductOfferMappingError, match="dictionary-like mapping"):
        map_product_offer(malformed_offer)


def test_contract_reports_malformed_required_external_field() -> None:
    """A malformed required value should identify its normalized contract field."""
    with pytest.raises(
        ProductOfferMappingError,
        match="product_name",
    ) as error_info:
        map_product_offer(
            {
                "product": ["Milk"],
                "store": "Market",
                "price": "3.99",
                "source": "weekly-flyer",
            }
        )

    assert isinstance(error_info.value.__cause__, ValidationError)
