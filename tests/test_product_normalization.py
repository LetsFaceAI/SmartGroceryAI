"""Focused tests for deterministic product-offer normalization."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.normalized_product import NormalizedProduct
from app.schemas.product_offer import MeasurementUnit, ProductOffer, PromotionStatus
from app.services.product_normalization import (
    ProductNormalizationError,
    normalize_product_name,
    normalize_product_offer,
    normalize_store_name,
    normalize_unit,
    parse_package_size,
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
    assert normalized.brand == "example dairy"
    assert normalized.store == "example grocer"
    assert normalized.package_quantity == 1
    assert normalized.package_size == Decimal("2000")
    assert normalized.total_package_size == Decimal("2000")
    assert normalized.unit == "mL"
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
    assert normalized.package_quantity is None
    assert normalized.package_size is None
    assert normalized.total_package_size is None
    assert normalized.unit is None
    assert normalized.regular_price is None


def test_normalization_does_not_convert_package_units() -> None:
    """Units without a simple scale conversion should remain exact."""
    offer = make_offer(package_size="500", unit=MeasurementUnit.GRAM)

    normalized = normalize_product_offer(offer)

    assert normalized.package_size == Decimal("500")
    assert normalized.total_package_size == Decimal("500")
    assert normalized.unit == "g"


def test_product_name_normalization_is_deterministic() -> None:
    """Equivalent Unicode/case/whitespace forms should produce the same key."""
    values = ["Café Milk", "ＣＡＦÉ   MILK", "  café\tmilk  "]

    assert {normalize_product_name(value) for value in values} == {"café milk"}


def test_product_name_normalization_removes_only_formatting_noise() -> None:
    """Trademark and edge decoration should disappear while details remain."""
    value = "  • ACME® 2% Lactose-Free Milk -  "

    assert normalize_product_name(value) == "acme 2% lactose-free milk"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Walmart Canada", "walmart"),
        ("Wal-Mart", "walmart"),
        ("RCSS", "real canadian superstore"),
        ("  Local   MARKET  ", "local market"),
    ],
)
def test_store_normalization_uses_explicit_aliases(
    value: str,
    expected: str,
) -> None:
    """Known aliases should converge while unknown stores receive text cleanup."""
    assert normalize_store_name(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("grams", MeasurementUnit.GRAM),
        ("KG", MeasurementUnit.KILOGRAM),
        ("ml", MeasurementUnit.MILLILITRE),
        ("L", MeasurementUnit.LITRE),
        ("lbs", MeasurementUnit.POUND),
        ("ct", MeasurementUnit.COUNT),
        ("pk", MeasurementUnit.PACK),
    ],
)
def test_unit_aliases_are_normalized(
    value: str,
    expected: MeasurementUnit,
) -> None:
    """Common spellings should resolve to the existing validated unit vocabulary."""
    assert normalize_unit(value) is expected


@pytest.mark.parametrize(
    ("value", "quantity", "size", "total", "unit"),
    [
        ("500 g", 1, Decimal("500"), Decimal("500"), "g"),
        ("1.5 L", 1, Decimal("1500.0"), Decimal("1500.0"), "mL"),
        ("12 pack", 1, Decimal("12"), Decimal("12"), "pack"),
        ("2 x 500 g", 2, Decimal("500"), Decimal("1000"), "g"),
        ("2×500g", 2, Decimal("500"), Decimal("1000"), "g"),
    ],
)
def test_simple_package_sizes_are_parsed_canonically(
    value: str,
    quantity: int,
    size: Decimal,
    total: Decimal,
    unit: str,
) -> None:
    """Supported single and multipack forms should have exact comparison values."""
    package = parse_package_size(value)

    assert package.package_quantity == quantity
    assert package.package_size == size
    assert package.total_package_size == total
    assert package.unit == unit


@pytest.mark.parametrize(
    "value",
    ["family size", "500", "about 500 g", "2 x g", "500 bananas", "0 g"],
)
def test_unsupported_package_sizes_fail_without_guessing(value: str) -> None:
    """Malformed, ambiguous, unknown, and non-positive packages should fail safely."""
    with pytest.raises(ProductNormalizationError):
        parse_package_size(value)


def test_unknown_unit_fails_without_guessing() -> None:
    """An unknown unit must not be silently treated as count or weight."""
    with pytest.raises(ProductNormalizationError, match="Unsupported measurement unit"):
        normalize_unit("bunches")


def test_normalized_product_rejects_unknown_fields() -> None:
    """Unexpected comparison fields should fail rather than silently disappear."""
    offer = make_offer()
    normalized_data = normalize_product_offer(offer).model_dump()
    normalized_data["matching_guess"] = "milk"

    with pytest.raises(ValidationError, match="matching_guess"):
        NormalizedProduct.model_validate(normalized_data)
