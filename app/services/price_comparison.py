"""Calculate and rank grocery unit prices with deterministic Decimal arithmetic."""

from collections.abc import Sequence
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal

from app.schemas.normalized_product import CanonicalUnit, NormalizedProduct
from app.schemas.price_comparison import (
    UNIT_PRICE_DECIMAL_PLACES,
    CheapestOfferSelection,
    CheapestOfferStatus,
    OfferPriceComparison,
    OfferValidityStatus,
    PriceComparisonStatus,
    UnitPriceUnit,
    determine_offer_validity,
)
from app.schemas.product_match import ProductMatchDecision, ProductMatchResult
from app.schemas.product_offer import MeasurementUnit, PriceBasis
from app.services.product_normalization import normalize_product_name

_UNIT_PRICE_QUANTUM = Decimal(1).scaleb(-UNIT_PRICE_DECIMAL_PLACES)

# Canonical metric inputs are scaled to familiar shelf-comparison denominators.
# Pack remains price-per-pack because unknown pack contents are not individual items.
_UNIT_PRICE_RULES: dict[CanonicalUnit, tuple[Decimal, UnitPriceUnit]] = {
    CanonicalUnit.GRAM: (Decimal("1000"), UnitPriceUnit.KILOGRAM),
    CanonicalUnit.MILLILITRE: (Decimal("1000"), UnitPriceUnit.LITRE),
    CanonicalUnit.EACH: (Decimal("1"), UnitPriceUnit.ITEM),
    CanonicalUnit.COUNT: (Decimal("1"), UnitPriceUnit.ITEM),
    CanonicalUnit.PACK: (Decimal("1"), UnitPriceUnit.PACK),
}


def _comparison(
    product: NormalizedProduct,
    *,
    as_of: date,
    validity_status: OfferValidityStatus,
    status: PriceComparisonStatus,
    reason: str,
    comparable_quantity: Decimal | None = None,
    comparable_unit: CanonicalUnit | None = None,
    unit_price: Decimal | None = None,
    unit_price_unit: UnitPriceUnit | None = None,
) -> OfferPriceComparison:
    """Build a comparison while consistently retaining provenance and date context."""
    return OfferPriceComparison(
        original_offer=product.original_offer,
        normalized_product=product,
        as_of=as_of,
        validity_status=validity_status,
        status=status,
        reason=reason,
        comparable_quantity=comparable_quantity,
        comparable_unit=comparable_unit,
        unit_price=unit_price,
        unit_price_unit=unit_price_unit,
        currency=product.currency,
    )


def _provider_basis_measurement(
    unit: MeasurementUnit,
) -> tuple[Decimal, CanonicalUnit]:
    """Convert a provider's one-unit price denominator into canonical measurement."""
    if unit is MeasurementUnit.KILOGRAM:
        return Decimal("1000"), CanonicalUnit.GRAM
    if unit is MeasurementUnit.LITRE:
        return Decimal("1000"), CanonicalUnit.MILLILITRE
    return Decimal("1"), CanonicalUnit(unit.value)


def calculate_unit_price(
    product: NormalizedProduct,
    *,
    as_of: date,
) -> OfferPriceComparison:
    """Calculate one active offer's unit price or an explicit non-comparable state."""
    offer = product.original_offer
    validity_status = determine_offer_validity(offer, as_of=as_of)
    if validity_status is not OfferValidityStatus.ACTIVE:
        return _comparison(
            product,
            as_of=as_of,
            validity_status=validity_status,
            status=PriceComparisonStatus.INELIGIBLE_VALIDITY,
            reason=(
                f"Offer validity is {validity_status.value} as of {as_of.isoformat()}; "
                "only active offers can be ranked as current deals."
            ),
        )

    if offer.price_basis is PriceBasis.UNKNOWN:
        return _comparison(
            product,
            as_of=as_of,
            validity_status=validity_status,
            status=PriceComparisonStatus.UNKNOWN_PRICE_BASIS,
            reason="The provider did not establish what quantity the price applies to.",
        )

    quantity: Decimal | None
    unit: CanonicalUnit | None
    if offer.price_basis in {PriceBasis.PER_WEIGHT, PriceBasis.PER_VOLUME}:
        # ProductOffer validation guarantees the correct measurement family here.
        assert offer.price_basis_unit is not None
        quantity, unit = _provider_basis_measurement(offer.price_basis_unit)
    else:
        quantity = product.total_package_size
        unit = product.unit
        if quantity is None or unit is None:
            if offer.price_basis is PriceBasis.EACH:
                quantity, unit = Decimal("1"), CanonicalUnit.EACH
            else:
                return _comparison(
                    product,
                    as_of=as_of,
                    validity_status=validity_status,
                    status=PriceComparisonStatus.MISSING_PACKAGE_DATA,
                    reason=(
                        "A total-package price requires package quantity and unit "
                        "for comparison."
                    ),
                )

    rule = _UNIT_PRICE_RULES.get(unit)
    if rule is None:
        return _comparison(
            product,
            as_of=as_of,
            validity_status=validity_status,
            status=PriceComparisonStatus.UNSUPPORTED_UNIT,
            reason=f"Unit '{unit.value}' has no deterministic comparison rule.",
            comparable_quantity=quantity,
            comparable_unit=unit,
        )

    scale, unit_price_unit = rule
    unit_price = ((offer.price * scale) / quantity).quantize(
        _UNIT_PRICE_QUANTUM,
        rounding=ROUND_HALF_EVEN,
    )
    return _comparison(
        product,
        as_of=as_of,
        validity_status=validity_status,
        status=PriceComparisonStatus.COMPARABLE,
        reason=f"Calculated price per {unit_price_unit.value}.",
        comparable_quantity=quantity,
        comparable_unit=unit,
        unit_price=unit_price,
        unit_price_unit=unit_price_unit,
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
    *,
    as_of: date,
) -> CheapestOfferSelection:
    """Rank only trusted, active, comparable offers with one currency/unit basis."""
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
    if match_results and any(
        result.request != match_results[0].request for result in match_results
    ):
        raise ValueError(
            "Every match result must preserve the same shopping constraints."
        )

    safe_results = [result for result in match_results if result.safe_for_ranking]
    uncertain = tuple(
        result
        for result in match_results
        if result.decision is ProductMatchDecision.UNCERTAIN
    )
    comparisons = tuple(
        calculate_unit_price(result.product, as_of=as_of) for result in safe_results
    )
    comparable = [
        comparison
        for comparison in comparisons
        if comparison.status is PriceComparisonStatus.COMPARABLE
    ]
    ignored_non_matches = sum(
        result.decision is ProductMatchDecision.NO_MATCH for result in match_results
    )

    if not comparable:
        return CheapestOfferSelection(
            requested_item_name=normalized_request,
            status=CheapestOfferStatus.NO_COMPARABLE_OFFERS,
            reason="No trusted match was both active and safely comparable.",
            comparisons=comparisons,
            ignored_non_matches=ignored_non_matches,
            uncertain_candidates=uncertain,
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
            uncertain_candidates=uncertain,
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
        uncertain_candidates=uncertain,
    )
