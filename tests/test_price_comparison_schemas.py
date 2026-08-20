"""Focused tests for deterministic price-comparison result contracts."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.normalized_product import CanonicalUnit, NormalizedProduct
from app.schemas.price_comparison import (
    OfferPriceComparison,
    OfferValidityStatus,
    PriceComparisonStatus,
    UnitPriceUnit,
)
from app.schemas.product_offer import MeasurementUnit, PriceBasis, ProductOffer
from app.services.product_normalization import normalize_product_offer

AS_OF = date(2026, 8, 20)


def make_product(
    *,
    package_size: str | None = "500",
    unit: MeasurementUnit | None = MeasurementUnit.GRAM,
) -> NormalizedProduct:
    """Create a validated normalized product with configurable package data."""
    offer = ProductOffer(
        product_name="Ground Coffee",
        store="Example Grocer",
        price="8.99",
        package_size=package_size,
        unit=unit,
        price_basis=PriceBasis.TOTAL_PACKAGE,
        valid_from=date(2026, 1, 1),
        valid_until=date(2026, 12, 31),
        source="fixture:coffee",
    )
    return normalize_product_offer(offer)


def test_comparable_result_retains_precise_unit_price() -> None:
    """A complete result should use Decimal for exact per-unit pricing."""
    product = make_product()

    result = OfferPriceComparison(
        original_offer=product.original_offer,
        normalized_product=product,
        as_of=AS_OF,
        validity_status=OfferValidityStatus.ACTIVE,
        status=PriceComparisonStatus.COMPARABLE,
        reason="The offer has a canonical mass quantity.",
        comparable_quantity=product.total_package_size,
        comparable_unit=product.unit,
        unit_price=Decimal("17.98"),
        unit_price_unit=UnitPriceUnit.KILOGRAM,
        currency="CAD",
    )

    assert result.comparable_quantity == Decimal("500")
    assert result.comparable_unit is CanonicalUnit.GRAM
    assert result.unit_price == Decimal("17.98")
    assert result.unit_price_unit is UnitPriceUnit.KILOGRAM
    assert isinstance(result.unit_price, Decimal)
    assert result.original_offer is product.original_offer


def test_missing_package_result_is_explicit() -> None:
    """An incomplete offer should remain valid without invented unit values."""
    product = make_product(package_size=None, unit=None)

    result = OfferPriceComparison(
        original_offer=product.original_offer,
        normalized_product=product,
        as_of=AS_OF,
        validity_status=OfferValidityStatus.ACTIVE,
        status=PriceComparisonStatus.MISSING_PACKAGE_DATA,
        reason="The flyer did not provide package size and unit.",
        currency="CAD",
    )

    assert result.comparable_quantity is None
    assert result.comparable_unit is None
    assert result.unit_price is None


def test_unsupported_unit_retains_known_measurement_without_unit_price() -> None:
    """Unsupported comparison policy should preserve data without guessing."""
    product = make_product(package_size="2", unit=MeasurementUnit.POUND)

    result = OfferPriceComparison(
        original_offer=product.original_offer,
        normalized_product=product,
        as_of=AS_OF,
        validity_status=OfferValidityStatus.ACTIVE,
        status=PriceComparisonStatus.UNSUPPORTED_UNIT,
        reason="Cross-system mass conversion is not enabled.",
        comparable_quantity=product.total_package_size,
        comparable_unit=product.unit,
        unit_price=None,
        currency="CAD",
    )

    assert result.comparable_quantity == Decimal("2")
    assert result.comparable_unit is CanonicalUnit.POUND
    assert result.unit_price is None


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"status": PriceComparisonStatus.COMPARABLE, "unit_price": None},
        {
            "status": PriceComparisonStatus.MISSING_PACKAGE_DATA,
            "comparable_quantity": Decimal("500"),
        },
        {
            "status": PriceComparisonStatus.UNSUPPORTED_UNIT,
            "unit_price": Decimal("0.01798"),
        },
    ],
)
def test_comparison_status_rejects_inconsistent_values(
    invalid_fields: dict[str, object],
) -> None:
    """Status must never contradict the comparison values exposed to callers."""
    product = make_product()
    comparison_data: dict[str, object] = {
        "original_offer": product.original_offer,
        "normalized_product": product,
        "as_of": AS_OF,
        "validity_status": OfferValidityStatus.ACTIVE,
        "status": PriceComparisonStatus.COMPARABLE,
        "reason": "Fixture comparison.",
        "comparable_quantity": Decimal("500"),
        "comparable_unit": CanonicalUnit.GRAM,
        "unit_price": Decimal("17.98"),
        "unit_price_unit": UnitPriceUnit.KILOGRAM,
        "currency": "CAD",
    }
    comparison_data.update(invalid_fields)

    with pytest.raises(ValidationError):
        OfferPriceComparison.model_validate(comparison_data)


def test_comparison_rejects_mismatched_source_offer() -> None:
    """A result cannot claim provenance from a different flyer offer."""
    product = make_product()
    other_offer = ProductOffer(
        product_name="Ground Coffee",
        store="Other Store",
        price="7.99",
        source="fixture:other-coffee",
    )

    with pytest.raises(ValidationError, match="original_offer must match"):
        OfferPriceComparison(
            original_offer=other_offer,
            normalized_product=product,
            as_of=AS_OF,
            validity_status=OfferValidityStatus.ACTIVE,
            status=PriceComparisonStatus.COMPARABLE,
            reason="Invalid mixed-source result.",
            comparable_quantity=Decimal("500"),
            comparable_unit=CanonicalUnit.GRAM,
            unit_price=Decimal("17.98"),
            unit_price_unit=UnitPriceUnit.KILOGRAM,
            currency="CAD",
        )


def test_comparison_rejects_validity_status_inconsistent_with_as_of() -> None:
    """Callers cannot label an expired offer active in a comparison contract."""
    product = make_product()
    comparison_data: dict[str, object] = {
        "original_offer": product.original_offer,
        "normalized_product": product,
        "as_of": date(2027, 1, 1),
        "validity_status": OfferValidityStatus.ACTIVE,
        "status": PriceComparisonStatus.COMPARABLE,
        "reason": "Invalid validity fixture.",
        "comparable_quantity": Decimal("500"),
        "comparable_unit": CanonicalUnit.GRAM,
        "unit_price": Decimal("17.98"),
        "unit_price_unit": UnitPriceUnit.KILOGRAM,
        "currency": "CAD",
    }

    with pytest.raises(ValidationError, match="validity_status must match"):
        OfferPriceComparison.model_validate(comparison_data)


@pytest.mark.parametrize(
    "incorrect_field",
    [
        {"comparable_quantity": Decimal("250")},
        {"comparable_unit": CanonicalUnit.MILLILITRE},
    ],
)
def test_comparison_rejects_measurement_not_from_normalized_product(
    incorrect_field: dict[str, object],
) -> None:
    """Comparison context cannot silently diverge from normalized package data."""
    product = make_product()
    comparison_data: dict[str, object] = {
        "original_offer": product.original_offer,
        "normalized_product": product,
        "as_of": AS_OF,
        "validity_status": OfferValidityStatus.ACTIVE,
        "status": PriceComparisonStatus.COMPARABLE,
        "reason": "Invalid measurement fixture.",
        "comparable_quantity": Decimal("500"),
        "comparable_unit": CanonicalUnit.GRAM,
        "unit_price": Decimal("17.98"),
        "unit_price_unit": UnitPriceUnit.KILOGRAM,
        "currency": "CAD",
    }
    comparison_data.update(incorrect_field)

    with pytest.raises(ValidationError):
        OfferPriceComparison.model_validate(comparison_data)
