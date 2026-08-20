"""Focused tests for deterministic product-offer normalization."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.normalized_product import NormalizedProduct
from app.schemas.product_offer import MeasurementUnit, ProductOffer, PromotionStatus
from app.services.product_normalization import (
    normalize_product_name,
    normalize_product_offer,
)


def make_offer(**overrides: object) -> ProductOffer:
    """Build one complete validated offer with concise per-test overrides."""
    offer_data: dict[str, object] = {
        "product_name": "Organic Whole Milk",
        "brand": "Example Dairy",
        "store": "Example Grocer",
        "package_size": "2",
        "unit": MeasurementUnit.LITRE,
        "price": "4.49",
        "regular_price": "5.49",
        "currency": "CAD",
        "promotion_status": PromotionStatus.SALE,
        "source": "fixture:milk-001",
    }
    offer_data.update(overrides)
    return ProductOffer.model_validate(offer_data)


def test_normalize_product_offer_returns_canonical_model() -> None:
    """The service should retain every initial price-comparison field."""
    offer = make_offer(product_name="  Organic   WHOLE Milk  ")

    normalized = normalize_product_offer(offer)

    assert type(normalized) is NormalizedProduct
    assert normalized.normalized_name == "organic whole milk"
    assert normalized.brand == "Example Dairy"
    assert normalized.store == "Example Grocer"
    assert normalized.package_size == Decimal("2")
    assert normalized.unit is MeasurementUnit.LITRE
    assert normalized.price == Decimal("4.49")
    assert normalized.regular_price == Decimal("5.49")
    assert normalized.currency == "CAD"
    assert normalized.promotion_status is PromotionStatus.SALE
    assert normalized.original_offer is offer


def test_normalization_preserves_missing_optional_values() -> None:
    """Normalization must not invent brand, package, or regular-price data."""
    offer = make_offer(
        brand=None,
        package_size=None,
        unit=None,
        regular_price=None,
        promotion_status=PromotionStatus.UNKNOWN,
    )

    normalized = normalize_product_offer(offer)

    assert normalized.brand is None
    assert normalized.package_size is None
    assert normalized.unit is None
    assert normalized.regular_price is None


def test_normalization_does_not_convert_package_units() -> None:
    """This phase should preserve package measurements exactly as validated."""
    offer = make_offer(package_size="500", unit=MeasurementUnit.GRAM)

    normalized = normalize_product_offer(offer)

    assert normalized.package_size == Decimal("500")
    assert normalized.unit is MeasurementUnit.GRAM


def test_product_name_normalization_is_deterministic() -> None:
    """Equivalent Unicode/case/whitespace forms should produce the same key."""
    values = ["Café Milk", "ＣＡＦÉ   MILK", "  café\tmilk  "]

    assert {normalize_product_name(value) for value in values} == {"café milk"}


def test_normalized_product_rejects_unknown_fields() -> None:
    """Unexpected comparison fields should fail rather than silently disappear."""
    offer = make_offer()
    normalized_data = normalize_product_offer(offer).model_dump()
    normalized_data["matching_guess"] = "milk"

    with pytest.raises(ValidationError, match="matching_guess"):
        NormalizedProduct.model_validate(normalized_data)
