"""Calculate and rank grocery unit prices with deterministic Decimal arithmetic."""

from collections.abc import Sequence
from decimal import ROUND_HALF_EVEN, Decimal

from app.schemas.normalized_product import CanonicalUnit, NormalizedProduct
from app.schemas.price_comparison import (
    UNIT_PRICE_DECIMAL_PLACES,
    CheapestOfferSelection,
    CheapestOfferStatus,
    OfferPriceComparison,
    PriceComparisonStatus,
    UnitPriceUnit,
)
from app.schemas.product_match import ProductMatchResult
from app.services.product_normalization import normalize_product_name

# Keep calculation precision aligned with the UnitPrice schema. A documented
# rounding policy makes repeating divisions stable across platforms and runs.
_UNIT_PRICE_QUANTUM = Decimal(1).scaleb(-UNIT_PRICE_DECIMAL_PLACES)

# Canonical metric inputs are scaled to familiar shelf-comparison denominators.
# Count and each share an item basis. Pack remains price-per-pack because treating
# unknown pack contents as individual items would be an unsafe assumption.
# Imperial units remain unsupported until an explicit conversion policy exists.
_UNIT_PRICE_RULES: dict[CanonicalUnit, tuple[Decimal, UnitPriceUnit]] = {
    CanonicalUnit.GRAM: (Decimal("1000"), UnitPriceUnit.KILOGRAM),
    CanonicalUnit.MILLILITRE: (Decimal("1000"), UnitPriceUnit.LITRE),
    CanonicalUnit.EACH: (Decimal("1"), UnitPriceUnit.ITEM),
    CanonicalUnit.COUNT: (Decimal("1"), UnitPriceUnit.ITEM),
    CanonicalUnit.PACK: (Decimal("1"), UnitPriceUnit.PACK),
}


def calculate_unit_price(product: NormalizedProduct) -> OfferPriceComparison:
    """Calculate one offer's unit price or return an explicit non-comparable state."""
    offer = product.original_offer
    quantity = product.total_package_size
    unit = product.unit
    if quantity is None or unit is None:
        return OfferPriceComparison(
            original_offer=offer,
            normalized_product=product,
            status=PriceComparisonStatus.MISSING_PACKAGE_DATA,
            reason="Package quantity and unit are required for unit-price calculation.",
            currency=offer.currency,
        )

    rule = _UNIT_PRICE_RULES.get(unit)
    if rule is None:
        return OfferPriceComparison(
            original_offer=offer,
            normalized_product=product,
            status=PriceComparisonStatus.UNSUPPORTED_UNIT,
            reason=f"Unit '{unit.value}' has no deterministic comparison rule.",
            comparable_quantity=quantity,
            comparable_unit=unit,
            currency=offer.currency,
        )

    scale, unit_price_unit = rule
    unit_price = ((offer.price * scale) / quantity).quantize(
        _UNIT_PRICE_QUANTUM,
        rounding=ROUND_HALF_EVEN,
    )
    return OfferPriceComparison(
        original_offer=offer,
        normalized_product=product,
        status=PriceComparisonStatus.COMPARABLE,
        reason=f"Calculated price per {unit_price_unit.value}.",
        comparable_quantity=quantity,
        comparable_unit=unit,
        unit_price=unit_price,
        unit_price_unit=unit_price_unit,
        currency=offer.currency,
    )


def _ranking_key(comparison: OfferPriceComparison) -> tuple[object, ...]:
    """Provide stable tie-breaking after exact unit price."""
    assert comparison.unit_price is not None
    offer = comparison.original_offer
    return (
        comparison.unit_price,
        offer.price,
        comparison.normalized_product.store,
        comparison.normalized_product.normalized_name,
        offer.source,
    )


def select_cheapest_offer(
    requested_item_name: str,
    match_results: Sequence[ProductMatchResult],
) -> CheapestOfferSelection:
    """Rank only matching, comparable offers with one currency/unit basis."""
    normalized_request = normalize_product_name(requested_item_name)
    if not any(character.isalnum() for character in normalized_request):
        raise ValueError(
            "requested_item_name must contain at least one letter or number."
        )
    if any(
        result.normalized_request_name != normalized_request for result in match_results
    ):
        raise ValueError(
            "Every match result must belong to the requested item being compared."
        )

    matching_results = [result for result in match_results if result.matched]
    comparisons = tuple(
        calculate_unit_price(result.product) for result in matching_results
    )
    comparable = [
        comparison
        for comparison in comparisons
        if comparison.status is PriceComparisonStatus.COMPARABLE
    ]
    ignored_non_matches = len(match_results) - len(matching_results)

    if not comparable:
        return CheapestOfferSelection(
            requested_item_name=normalized_request,
            status=CheapestOfferStatus.NO_COMPARABLE_OFFERS,
            reason="No matching offer had complete, supported package data.",
            comparisons=comparisons,
            ignored_non_matches=ignored_non_matches,
        )

    comparison_groups = {
        (comparison.currency, comparison.unit_price_unit) for comparison in comparable
    }
    if len(comparison_groups) != 1:
        return CheapestOfferSelection(
            requested_item_name=normalized_request,
            status=CheapestOfferStatus.INCOMPATIBLE_COMPARISON_GROUPS,
            reason="Comparable offers did not share one currency and unit-price basis.",
            comparisons=comparisons,
            ignored_non_matches=ignored_non_matches,
        )

    ranked = tuple(sorted(comparable, key=_ranking_key))
    cheapest = ranked[0]
    tied = tuple(
        comparison
        for comparison in ranked
        if comparison.unit_price == cheapest.unit_price
    )
    return CheapestOfferSelection(
        requested_item_name=normalized_request,
        status=CheapestOfferStatus.SELECTED,
        reason="Selected the lowest exact unit price using deterministic tie-breakers.",
        comparisons=comparisons,
        ranked_comparable_offers=ranked,
        cheapest_offer=cheapest,
        tied_cheapest_offers=tied,
        ignored_non_matches=ignored_non_matches,
    )
