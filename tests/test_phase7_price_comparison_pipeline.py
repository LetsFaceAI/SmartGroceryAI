"""Integration tests for the deterministic Phase 7 comparison pipeline.

Every test uses real validation, normalization, matching, calculation, and
selection code. No external service or probabilistic component is involved.
"""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.price_comparison import (
    CheapestOfferSelection,
    CheapestOfferStatus,
    PriceComparisonStatus,
    UnitPriceUnit,
)
from app.schemas.product_match import ProductMatchResult
from app.schemas.product_offer import MeasurementUnit, PriceBasis, ProductOffer
from app.services.price_comparison import (
    calculate_unit_price,
    select_cheapest_offer,
)
from app.services.product_matcher import match_product
from app.services.product_normalization import normalize_product_offer

AS_OF = date(2026, 8, 20)


def make_match(
    *,
    requested_name: str = "ground coffee",
    product_name: str = "Ground Coffee",
    store: str = "Example Grocer",
    price: str = "8.00",
    package_size: str | None = "1",
    package_quantity: int = 1,
    unit: MeasurementUnit | None = MeasurementUnit.KILOGRAM,
    currency: str = "CAD",
    source: str = "fixture:coffee",
) -> ProductMatchResult:
    """Build one candidate through every Phase 6 production boundary."""
    offer = ProductOffer(
        product_name=product_name,
        store=store,
        price=price,
        package_size=package_size,
        package_quantity=package_quantity,
        unit=unit,
        price_basis=PriceBasis.TOTAL_PACKAGE,
        valid_from=date(2026, 1, 1),
        valid_until=date(2026, 12, 31),
        currency=currency,
        source=source,
    )
    return match_product(requested_name, normalize_product_offer(offer))


@pytest.mark.parametrize(
    ("package_size", "unit", "price", "expected_price", "expected_unit"),
    [
        ("500", MeasurementUnit.GRAM, "4.00", "8.000000000000", UnitPriceUnit.KILOGRAM),
        (
            "1.5",
            MeasurementUnit.KILOGRAM,
            "6.00",
            "4.000000000000",
            UnitPriceUnit.KILOGRAM,
        ),
        (
            "500",
            MeasurementUnit.MILLILITRE,
            "2.00",
            "4.000000000000",
            UnitPriceUnit.LITRE,
        ),
        ("2", MeasurementUnit.LITRE, "3.00", "1.500000000000", UnitPriceUnit.LITRE),
        ("12", MeasurementUnit.COUNT, "3.60", "0.300000000000", UnitPriceUnit.ITEM),
        ("1", MeasurementUnit.EACH, "2.25", "2.250000000000", UnitPriceUnit.ITEM),
        ("6", MeasurementUnit.PACK, "6.00", "1.000000000000", UnitPriceUnit.PACK),
    ],
)
def test_unit_price_rules_use_exact_decimal_arithmetic(
    package_size: str,
    unit: MeasurementUnit,
    price: str,
    expected_price: str,
    expected_unit: UnitPriceUnit,
) -> None:
    """Metric scale variants and discrete counts should share stable denominators."""
    match = make_match(package_size=package_size, unit=unit, price=price)

    comparison = calculate_unit_price(match.product, as_of=AS_OF)

    assert comparison.status is PriceComparisonStatus.COMPARABLE
    assert comparison.unit_price == Decimal(expected_price)
    assert comparison.unit_price_unit is expected_unit
    assert isinstance(comparison.unit_price, Decimal)


def test_different_package_sizes_select_lowest_price_per_kilogram() -> None:
    """A larger shelf price may still be the cheapest normalized offer."""
    small = make_match(
        store="Corner Market",
        price="4.00",
        package_size="500",
        unit=MeasurementUnit.GRAM,
        source="fixture:small-coffee",
    )
    large = make_match(
        store="Warehouse Market",
        price="7.00",
        package_size="1",
        unit=MeasurementUnit.KILOGRAM,
        source="fixture:large-coffee",
    )

    selection = select_cheapest_offer("ground coffee", [small, large], as_of=AS_OF)

    assert selection.status is CheapestOfferStatus.SELECTED
    assert selection.cheapest_offer is not None
    assert selection.cheapest_offer.original_offer.store == "Warehouse Market"
    assert selection.cheapest_offer.unit_price == Decimal("7.000000000000")
    assert [offer.unit_price for offer in selection.ranked_comparable_offers] == [
        Decimal("7.000000000000"),
        Decimal("8.000000000000"),
    ]


def test_non_comparable_matches_do_not_hide_valid_offer() -> None:
    """Missing and unsupported candidates should be reported but never ranked."""
    valid = make_match(store="Valid Store", source="fixture:valid")
    missing = make_match(
        store="Missing Store",
        package_size=None,
        unit=None,
        source="fixture:missing",
    )
    unsupported = make_match(
        store="Imperial Store",
        package_size="2",
        unit=MeasurementUnit.POUND,
        source="fixture:unsupported",
    )

    selection = select_cheapest_offer(
        "ground coffee",
        [unsupported, missing, valid],
        as_of=AS_OF,
    )

    assert selection.status is CheapestOfferStatus.SELECTED
    assert selection.cheapest_offer is not None
    assert selection.cheapest_offer.original_offer.store == "Valid Store"
    assert {comparison.status for comparison in selection.comparisons} == {
        PriceComparisonStatus.COMPARABLE,
        PriceComparisonStatus.MISSING_PACKAGE_DATA,
        PriceComparisonStatus.UNSUPPORTED_UNIT,
    }
    assert len(selection.ranked_comparable_offers) == 1


def test_no_comparable_offer_returns_clear_status() -> None:
    """Selection should remain empty when every matching offer lacks a safe rule."""
    missing = make_match(package_size=None, unit=None, source="fixture:missing")
    unsupported = make_match(
        package_size="12",
        unit=MeasurementUnit.OUNCE,
        source="fixture:ounces",
    )

    selection = select_cheapest_offer(
        "ground coffee", [missing, unsupported], as_of=AS_OF
    )

    assert selection.status is CheapestOfferStatus.NO_COMPARABLE_OFFERS
    assert selection.cheapest_offer is None
    assert selection.ranked_comparable_offers == ()


def test_equal_unit_prices_return_ties_with_deterministic_winner() -> None:
    """All exact ties remain visible while lower shelf price breaks selection ties."""
    small = make_match(
        store="Beta Market",
        price="4.00",
        package_size="500",
        unit=MeasurementUnit.GRAM,
        source="fixture:beta-small",
    )
    large = make_match(
        store="Alpha Market",
        price="8.00",
        package_size="1",
        unit=MeasurementUnit.KILOGRAM,
        source="fixture:alpha-large",
    )

    selection = select_cheapest_offer("ground coffee", [large, small], as_of=AS_OF)

    assert selection.cheapest_offer is not None
    assert selection.cheapest_offer.original_offer.store == "Beta Market"
    assert len(selection.tied_cheapest_offers) == 2
    assert {
        comparison.original_offer.store for comparison in selection.tied_cheapest_offers
    } == {"Alpha Market", "Beta Market"}


def test_selection_schema_rejects_false_tie_group() -> None:
    """A result cannot label a more expensive ranked offer as an exact tie."""
    cheaper = make_match(price="7.00", source="fixture:cheaper")
    expensive = make_match(price="8.00", source="fixture:expensive")
    selection = select_cheapest_offer(
        "ground coffee", [cheaper, expensive], as_of=AS_OF
    )
    selection_data = selection.model_dump()
    selection_data["tied_cheapest_offers"] = [selection.ranked_comparable_offers[-1]]

    with pytest.raises(ValidationError, match="must share the cheapest"):
        CheapestOfferSelection.model_validate(selection_data)


def test_non_matching_offer_is_ignored_before_price_calculation() -> None:
    """A numerically cheap unrelated product must never enter grocery ranking."""
    matching = make_match(store="Coffee Store", source="fixture:coffee")
    unrelated = make_match(
        requested_name="ground coffee",
        product_name="Orange Juice",
        store="Juice Store",
        price="0.50",
        package_size="2",
        unit=MeasurementUnit.LITRE,
        source="fixture:juice",
    )

    selection = select_cheapest_offer(
        "ground coffee", [unrelated, matching], as_of=AS_OF
    )

    assert unrelated.matched is False
    assert selection.ignored_non_matches == 1
    assert len(selection.comparisons) == 1
    assert selection.cheapest_offer is not None
    assert selection.cheapest_offer.original_offer.store == "Coffee Store"


def test_selection_rejects_match_results_for_another_request() -> None:
    """A valid match from another grocery request cannot enter this ranking."""
    milk_match = make_match(
        requested_name="milk",
        product_name="Milk",
        source="fixture:milk",
    )

    with pytest.raises(ValueError, match="must belong to the requested item"):
        select_cheapest_offer("ground coffee", [milk_match], as_of=AS_OF)


def test_incompatible_units_or_currencies_are_not_ranked_together() -> None:
    """A numeric minimum is meaningless across different bases or currencies."""
    mass = make_match(source="fixture:mass-cad")
    count = make_match(
        price="6.00",
        package_size="12",
        unit=MeasurementUnit.COUNT,
        source="fixture:count-cad",
    )

    selection = select_cheapest_offer("ground coffee", [mass, count], as_of=AS_OF)

    assert selection.status is CheapestOfferStatus.INCOMPATIBLE_COMPARISON_GROUPS
    assert selection.cheapest_offer is None
    assert selection.ranked_comparable_offers == ()


def test_different_currencies_are_not_ranked_together() -> None:
    """Unit prices require one currency even when their measurement basis matches."""
    canadian = make_match(currency="CAD", source="fixture:coffee-cad")
    american = make_match(currency="USD", source="fixture:coffee-usd")

    selection = select_cheapest_offer(
        "ground coffee", [canadian, american], as_of=AS_OF
    )

    assert selection.status is CheapestOfferStatus.INCOMPATIBLE_COMPARISON_GROUPS
    assert selection.cheapest_offer is None


def test_pack_and_item_counts_are_not_assumed_equivalent() -> None:
    """Unknown pack contents cannot safely be relabeled as individual items."""
    counted_items = make_match(
        package_size="6",
        unit=MeasurementUnit.COUNT,
        source="fixture:counted-items",
    )
    packs = make_match(
        package_size="6",
        unit=MeasurementUnit.PACK,
        source="fixture:packs",
    )

    selection = select_cheapest_offer(
        "ground coffee", [counted_items, packs], as_of=AS_OF
    )

    assert selection.status is CheapestOfferStatus.INCOMPATIBLE_COMPARISON_GROUPS
    assert selection.cheapest_offer is None


def test_invalid_offer_fails_before_it_can_enter_selection() -> None:
    """Pydantic should reject invalid prices before normalization or ranking."""
    with pytest.raises(ValidationError, match="price"):
        ProductOffer(
            product_name="Ground Coffee",
            store="Invalid Store",
            price="0",
            package_size="1",
            unit=MeasurementUnit.KILOGRAM,
            source="fixture:invalid",
        )
