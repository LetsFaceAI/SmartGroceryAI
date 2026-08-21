"""Regression tests for immutable offer data retained by downstream snapshots."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.product_offer import MeasurementUnit, PriceBasis, ProductOffer
from app.services.price_comparison import calculate_unit_price
from app.services.product_normalization import normalize_product_offer


def test_offer_cannot_be_mutated_after_normalization_and_comparison() -> None:
    """Every retained reference must continue to satisfy its original validation."""
    offer = ProductOffer(
        product_name="Ground Coffee",
        store="Example Grocer",
        price=Decimal("8.00"),
        package_size=Decimal("1"),
        unit=MeasurementUnit.KILOGRAM,
        price_basis=PriceBasis.TOTAL_PACKAGE,
        valid_from=date(2026, 8, 1),
        valid_until=date(2026, 8, 31),
        source="fixture:immutable-coffee",
    )
    product = normalize_product_offer(offer)
    comparison = calculate_unit_price(product, as_of=date(2026, 8, 20))

    with pytest.raises(ValidationError, match="frozen"):
        offer.price = Decimal("0")
    with pytest.raises(ValidationError, match="frozen"):
        product.original_offer.valid_until = date(2026, 7, 1)
    with pytest.raises(ValidationError, match="frozen"):
        comparison.original_offer.price_basis = PriceBasis.UNKNOWN

    assert product.original_offer.price == Decimal("8.00")
    assert comparison.original_offer.valid_until == date(2026, 8, 31)
    assert comparison.unit_price == Decimal("8.000000000000")
