"""Focused tests for deterministic price-comparison result contracts."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.normalized_product import CanonicalUnit, NormalizedProduct
from app.schemas.price_comparison import (
    CheapestOfferSelection,
    CheapestOfferStatus,
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
    price: str = "8.99",
    store: str = "Example Grocer",
) -> NormalizedProduct:
    """Create a validated normalized product with configurable package data."""
    offer = ProductOffer(
        product_name="Ground Coffee",
        store=store,
        price=price,
        package_size=package_size,
        unit=unit,
        price_basis=PriceBasis.TOTAL_PACKAGE,
        valid_from=date(2026, 1, 1),
        valid_until=date(2026, 12, 31),
        source=f"fixture:coffee:{store}",
    )
    return normalize_product_offer(offer)


def make_comparison(*, store: str, unit_price: str) -> OfferPriceComparison:
    """Create one internally valid comparable result for selection-schema tests."""
    product = make_product(
        package_size="1",
        unit=MeasurementUnit.KILOGRAM,
        price=unit_price,
        store=store,
    )
    return OfferPriceComparison(
        original_offer=product.original_offer,
        normalized_product=product,
        as_of=AS_OF,
        validity_status=OfferValidityStatus.ACTIVE,
        status=PriceComparisonStatus.COMPARABLE,
        reason="Valid mass comparison fixture.",
        comparable_quantity=product.total_package_size,
        comparable_unit=product.unit,
        unit_price=Decimal(unit_price),
        unit_price_unit=UnitPriceUnit.KILOGRAM,
        currency="CAD",
    )


def selection_data(
    ranked: tuple[OfferPriceComparison, ...],
    ties: tuple[OfferPriceComparison, ...],
) -> dict[str, object]:
    """Build direct selection input without relying on the selection service."""
    return {
        "requested_item_name": "ground coffee",
        "status": CheapestOfferStatus.SELECTED,
        "reason": "Direct schema fixture.",
        "comparisons": ranked,
        "ranked_comparable_offers": ranked,
        "cheapest_offer": ranked[0],
        "tied_cheapest_offers": ties,
    }


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


@pytest.mark.parametrize(
    ("measurement_unit", "expected_comparable_unit", "unit_price_unit"),
    [
        (
            MeasurementUnit.GRAM,
            CanonicalUnit.GRAM,
            UnitPriceUnit.KILOGRAM,
        ),
        (
            MeasurementUnit.MILLILITRE,
            CanonicalUnit.MILLILITRE,
            UnitPriceUnit.LITRE,
        ),
        (MeasurementUnit.EACH, CanonicalUnit.EACH, UnitPriceUnit.ITEM),
        (MeasurementUnit.COUNT, CanonicalUnit.COUNT, UnitPriceUnit.ITEM),
        (MeasurementUnit.PACK, CanonicalUnit.PACK, UnitPriceUnit.PACK),
    ],
)
def test_comparable_unit_accepts_only_its_valid_unit_price_basis(
    measurement_unit: MeasurementUnit,
    expected_comparable_unit: CanonicalUnit,
    unit_price_unit: UnitPriceUnit,
) -> None:
    """Every supported measurement family should accept its canonical denominator."""
    product = make_product(package_size="1", unit=measurement_unit)

    comparison = OfferPriceComparison(
        original_offer=product.original_offer,
        normalized_product=product,
        as_of=AS_OF,
        validity_status=OfferValidityStatus.ACTIVE,
        status=PriceComparisonStatus.COMPARABLE,
        reason="Valid unit-basis fixture.",
        comparable_quantity=product.total_package_size,
        comparable_unit=product.unit,
        unit_price=Decimal("1.00"),
        unit_price_unit=unit_price_unit,
        currency="CAD",
    )

    assert comparison.comparable_unit is expected_comparable_unit
    assert comparison.unit_price_unit is unit_price_unit


@pytest.mark.parametrize(
    ("measurement_unit", "invalid_unit_price_unit"),
    [
        (MeasurementUnit.GRAM, UnitPriceUnit.ITEM),
        (MeasurementUnit.MILLILITRE, UnitPriceUnit.ITEM),
        (MeasurementUnit.PACK, UnitPriceUnit.KILOGRAM),
        (MeasurementUnit.COUNT, UnitPriceUnit.LITRE),
    ],
)
def test_comparable_unit_rejects_invalid_unit_price_basis(
    measurement_unit: MeasurementUnit,
    invalid_unit_price_unit: UnitPriceUnit,
) -> None:
    """Mass, volume, count, and pack units cannot use another family's basis."""
    product = make_product(package_size="1", unit=measurement_unit)

    with pytest.raises(ValidationError, match="invalid for the comparable_unit"):
        OfferPriceComparison(
            original_offer=product.original_offer,
            normalized_product=product,
            as_of=AS_OF,
            validity_status=OfferValidityStatus.ACTIVE,
            status=PriceComparisonStatus.COMPARABLE,
            reason="Invalid unit-basis fixture.",
            comparable_quantity=product.total_package_size,
            comparable_unit=product.unit,
            unit_price=Decimal("1.00"),
            unit_price_unit=invalid_unit_price_unit,
            currency="CAD",
        )


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


def test_selection_rejects_later_ranked_offer_with_lower_price() -> None:
    """The first ranked offer cannot claim cheapest when a later offer costs less."""
    first = make_comparison(store="First Store", unit_price="5.00")
    later_cheaper = make_comparison(store="Later Store", unit_price="4.00")

    with pytest.raises(ValidationError, match="nondecreasing unit_price"):
        CheapestOfferSelection.model_validate(
            selection_data((first, later_cheaper), (first,))
        )


def test_selection_rejects_tie_group_missing_cheapest_offer() -> None:
    """The tie group must begin with and contain the selected cheapest offer."""
    cheapest = make_comparison(store="Alpha Store", unit_price="4.00")
    tied = make_comparison(store="Beta Store", unit_price="4.00")

    with pytest.raises(ValidationError, match="exactly match the minimum-price"):
        CheapestOfferSelection.model_validate(selection_data((cheapest, tied), (tied,)))


def test_selection_rejects_incomplete_minimum_price_tie_group() -> None:
    """Every offer sharing the minimum price must appear in the tie group."""
    cheapest = make_comparison(store="Alpha Store", unit_price="4.00")
    tied = make_comparison(store="Beta Store", unit_price="4.00")
    higher = make_comparison(store="Gamma Store", unit_price="5.00")

    with pytest.raises(ValidationError, match="exactly match the minimum-price"):
        CheapestOfferSelection.model_validate(
            selection_data((cheapest, tied, higher), (cheapest,))
        )


def test_selection_accepts_complete_tie_group_in_ranked_order() -> None:
    """A complete ordered minimum-price group should satisfy the selection contract."""
    cheapest = make_comparison(store="Alpha Store", unit_price="4.00")
    tied = make_comparison(store="Beta Store", unit_price="4.00")
    higher = make_comparison(store="Gamma Store", unit_price="5.00")

    selection = CheapestOfferSelection.model_validate(
        selection_data((cheapest, tied, higher), (cheapest, tied))
    )

    assert selection.cheapest_offer == cheapest
    assert selection.tied_cheapest_offers == (cheapest, tied)
