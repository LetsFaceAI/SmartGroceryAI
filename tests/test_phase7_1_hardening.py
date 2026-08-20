"""Regression coverage for constraint, validity, and price-basis hardening."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.price_comparison import (
    CheapestOfferStatus,
    OfferValidityStatus,
    PriceComparisonStatus,
    UnitPriceUnit,
)
from app.schemas.product_match import ProductMatchDecision, ProductMatchType
from app.schemas.product_offer import (
    MeasurementUnit,
    PriceBasis,
    ProductOffer,
)
from app.schemas.shopping import (
    ConstraintRequirement,
    ShoppingConstraint,
    ShoppingItem,
)
from app.services.price_comparison import (
    calculate_unit_price,
    determine_offer_validity,
    select_cheapest_offer,
)
from app.services.product_matcher import build_product_match_request, match_product
from app.services.product_normalization import normalize_product_offer

AS_OF = date(2026, 8, 20)


def make_offer(
    product_name: str,
    *,
    price: str = "4.00",
    price_basis: PriceBasis = PriceBasis.TOTAL_PACKAGE,
    price_basis_unit: MeasurementUnit | None = None,
    package_size: str | None = "1",
    unit: MeasurementUnit | None = MeasurementUnit.EACH,
    valid_from: date | None = date(2026, 8, 1),
    valid_until: date | None = date(2026, 8, 31),
    store: str = "Example Grocer",
) -> ProductOffer:
    """Create a small active offer while allowing one policy field to vary."""
    return ProductOffer(
        product_name=product_name,
        store=store,
        price=price,
        price_basis=price_basis,
        price_basis_unit=price_basis_unit,
        package_size=package_size,
        unit=unit,
        valid_from=valid_from,
        valid_until=valid_until,
        source=f"fixture:{store}:{product_name}",
    )


def test_match_request_preserves_item_context_and_constraint_strength() -> None:
    """Notes, global preferences, and required/optional qualifiers must survive."""
    item = ShoppingItem(
        name="milk",
        quantity=2,
        unit="cartons",
        notes="Must be lactose free and 2% for dietary needs",
        constraints=[
            ShoppingConstraint(
                value="organic",
                requirement=ConstraintRequirement.OPTIONAL,
            )
        ],
    )

    request = build_product_match_request(
        item,
        request_preferences=["gluten-free when available", "lowest price"],
    )

    assert request.item_name == "milk"
    assert request.notes == item.notes
    assert request.shopping_item == item
    assert request.shopping_item.quantity == 2
    assert request.shopping_item.unit == "cartons"
    assert request.request_preferences == (
        "gluten-free when available",
        "lowest price",
    )
    assert {(item.value, item.requirement) for item in request.constraints} == {
        ("organic", ConstraintRequirement.OPTIONAL),
        ("2%", ConstraintRequirement.REQUIRED),
        ("lactose-free", ConstraintRequirement.REQUIRED),
        ("gluten-free", ConstraintRequirement.OPTIONAL),
    }


def test_optional_legacy_note_does_not_become_required() -> None:
    """Explicit optional language in old notes must not become a hard filter."""
    request = build_product_match_request(
        ShoppingItem(name="milk", notes="Lactose-free if available")
    )

    assert request.constraints == (
        ShoppingConstraint(
            value="lactose-free",
            requirement=ConstraintRequirement.OPTIONAL,
        ),
    )


def test_explicit_optional_constraint_wins_over_inferred_note_text() -> None:
    """Legacy note inference cannot strengthen an explicit user preference."""
    request = build_product_match_request(
        ShoppingItem(
            name="milk",
            notes="Organic if available",
            constraints=[
                ShoppingConstraint(
                    value="organic",
                    requirement=ConstraintRequirement.OPTIONAL,
                )
            ],
        )
    )

    assert request.constraints[0].requirement is ConstraintRequirement.OPTIONAL


def test_mixed_legacy_note_keeps_requirement_strength_per_clause() -> None:
    """An optional clause must not weaken a separate mandatory qualifier."""
    request = build_product_match_request(
        ShoppingItem(
            name="milk",
            notes="Must be lactose-free, but prefer organic if available",
        )
    )

    assert {(item.value, item.requirement) for item in request.constraints} == {
        ("lactose-free", ConstraintRequirement.REQUIRED),
        ("organic", ConstraintRequirement.OPTIONAL),
    }


def test_missing_required_constraint_is_uncertain_not_equivalent() -> None:
    """A name match cannot silently discard a required organic qualifier."""
    request = build_product_match_request(
        ShoppingItem(
            name="milk",
            constraints=[ShoppingConstraint(value="organic")],
        )
    )
    product = normalize_product_offer(make_offer("2% Milk"))

    result = match_product(request, product)

    assert result.decision is ProductMatchDecision.UNCERTAIN
    assert result.matched is False
    assert result.candidate is True
    assert result.safe_for_ranking is False
    assert result.unmet_required_constraints == ("organic",)


@pytest.mark.parametrize("constraint", ["organic", "2%", "lactose-free", "gluten-free"])
def test_protected_required_qualifiers_are_never_silently_dropped(
    constraint: str,
) -> None:
    """Every protected qualifier blocks ranking when the offer does not prove it."""
    request = build_product_match_request(
        ShoppingItem(
            name="milk",
            constraints=[ShoppingConstraint(value=constraint)],
        )
    )

    result = match_product(request, normalize_product_offer(make_offer("Milk")))

    assert result.decision is ProductMatchDecision.UNCERTAIN
    assert result.unmet_required_constraints == (constraint,)


def test_missing_optional_constraint_does_not_block_safe_match() -> None:
    """An unmet preference stays visible without becoming a mandatory filter."""
    request = build_product_match_request(
        ShoppingItem(
            name="milk",
            constraints=[
                ShoppingConstraint(
                    value="organic",
                    requirement=ConstraintRequirement.OPTIONAL,
                )
            ],
        )
    )

    result = match_product(request, normalize_product_offer(make_offer("Milk")))

    assert result.decision is ProductMatchDecision.SAFE
    assert result.safe_for_ranking is True
    assert result.unmet_optional_constraints == ("organic",)


@pytest.mark.parametrize(
    ("requested_name", "product_name"),
    [("milk", "Milk Chocolate"), ("apple", "Apple Juice")],
)
def test_category_changing_containment_is_only_an_uncertain_candidate(
    requested_name: str,
    product_name: str,
) -> None:
    """Containment must not equate a base grocery with a derived product category."""
    result = match_product(
        requested_name,
        normalize_product_offer(make_offer(product_name)),
    )

    assert result.match_type is ProductMatchType.CONTAINMENT
    assert result.decision is ProductMatchDecision.UNCERTAIN
    assert result.matched is False
    assert result.candidate is True
    assert result.safe_for_ranking is False


def test_uncertain_candidate_is_exposed_but_not_price_ranked() -> None:
    """A cheap category-changing candidate cannot beat a trusted milk offer."""
    uncertain = match_product(
        "milk",
        normalize_product_offer(make_offer("Milk Chocolate", price="0.50")),
    )
    safe = match_product(
        "milk",
        normalize_product_offer(make_offer("Milk", price="3.00")),
    )

    selection = select_cheapest_offer("milk", [uncertain, safe], as_of=AS_OF)

    assert selection.status is CheapestOfferStatus.SELECTED
    assert selection.cheapest_offer is not None
    assert selection.cheapest_offer.original_offer.product_name == "Milk"
    assert selection.uncertain_candidates == (uncertain,)


def test_selection_rejects_results_with_different_constraint_context() -> None:
    """Same-name results from different user requirements cannot be mixed."""
    unconstrained = match_product(
        "milk",
        normalize_product_offer(make_offer("Milk")),
    )
    constrained_request = build_product_match_request(
        ShoppingItem(
            name="milk",
            constraints=[ShoppingConstraint(value="organic")],
        )
    )
    constrained = match_product(
        constrained_request,
        normalize_product_offer(make_offer("Organic Milk")),
    )

    with pytest.raises(ValueError, match="same shopping constraints"):
        select_cheapest_offer(
            "milk",
            [unconstrained, constrained],
            as_of=AS_OF,
        )


@pytest.mark.parametrize(
    ("valid_from", "valid_until", "expected"),
    [
        (date(2026, 8, 1), date(2026, 8, 31), OfferValidityStatus.ACTIVE),
        (date(2026, 8, 21), date(2026, 8, 31), OfferValidityStatus.UPCOMING),
        (date(2026, 8, 1), date(2026, 8, 19), OfferValidityStatus.EXPIRED),
        (None, None, OfferValidityStatus.UNKNOWN),
    ],
)
def test_offer_validity_uses_explicit_as_of_date(
    valid_from: date | None,
    valid_until: date | None,
    expected: OfferValidityStatus,
) -> None:
    """Validity classification must not depend on the machine's current date."""
    offer = make_offer(
        "Milk",
        valid_from=valid_from,
        valid_until=valid_until,
    )

    assert determine_offer_validity(offer, as_of=AS_OF) is expected


@pytest.mark.parametrize(
    ("valid_from", "valid_until"),
    [(date(2026, 8, 1), None), (None, date(2026, 8, 31))],
)
def test_partial_current_window_remains_unknown(
    valid_from: date | None,
    valid_until: date | None,
) -> None:
    """A single non-decisive date cannot prove that an offer is currently active."""
    offer = make_offer("Milk", valid_from=valid_from, valid_until=valid_until)

    assert determine_offer_validity(offer, as_of=AS_OF) is OfferValidityStatus.UNKNOWN


def test_expired_offer_is_reported_but_not_ranked_as_current() -> None:
    """Expired pricing remains auditable while an active offer wins selection."""
    expired = match_product(
        "milk",
        normalize_product_offer(
            make_offer(
                "Milk",
                price="1.00",
                valid_until=date(2026, 8, 19),
            )
        ),
    )
    active = match_product(
        "milk",
        normalize_product_offer(make_offer("Milk", price="3.00", store="Active Store")),
    )

    selection = select_cheapest_offer("milk", [expired, active], as_of=AS_OF)

    assert selection.cheapest_offer is not None
    assert selection.cheapest_offer.original_offer.store == "Active Store"
    assert {comparison.validity_status for comparison in selection.comparisons} == {
        OfferValidityStatus.ACTIVE,
        OfferValidityStatus.EXPIRED,
    }


def test_unknown_validity_is_not_ranked_as_a_current_deal() -> None:
    """Absent flyer dates must remain unknown instead of being assumed active."""
    product = normalize_product_offer(
        make_offer("Milk", valid_from=None, valid_until=None)
    )

    comparison = calculate_unit_price(product, as_of=AS_OF)

    assert comparison.validity_status is OfferValidityStatus.UNKNOWN
    assert comparison.status is PriceComparisonStatus.INELIGIBLE_VALIDITY
    assert comparison.unit_price is None


def test_unknown_price_basis_is_non_comparable_even_with_package_data() -> None:
    """Package fields cannot prove whether an ambiguous advertised price is total."""
    product = normalize_product_offer(
        make_offer("Milk", price_basis=PriceBasis.UNKNOWN)
    )

    comparison = calculate_unit_price(product, as_of=AS_OF)

    assert comparison.status is PriceComparisonStatus.UNKNOWN_PRICE_BASIS
    assert comparison.unit_price is None


def test_each_price_without_package_data_compares_per_item() -> None:
    """An explicit each qualifier provides a safe one-item denominator."""
    product = normalize_product_offer(
        make_offer(
            "Cantaloupe",
            price="2.99",
            price_basis=PriceBasis.EACH,
            package_size=None,
            unit=None,
        )
    )

    comparison = calculate_unit_price(product, as_of=AS_OF)

    assert comparison.status is PriceComparisonStatus.COMPARABLE
    assert comparison.unit_price == Decimal("2.990000000000")
    assert comparison.unit_price_unit is UnitPriceUnit.ITEM


def test_per_kilogram_price_is_not_divided_by_package_size() -> None:
    """A per-weight advertisement is already a unit price, not a package total."""
    product = normalize_product_offer(
        make_offer(
            "Ground Beef",
            price="6.99",
            price_basis=PriceBasis.PER_WEIGHT,
            price_basis_unit=MeasurementUnit.KILOGRAM,
            package_size=None,
            unit=None,
        )
    )

    comparison = calculate_unit_price(product, as_of=AS_OF)

    assert comparison.status is PriceComparisonStatus.COMPARABLE
    assert comparison.unit_price == Decimal("6.990000000000")
    assert comparison.unit_price_unit is UnitPriceUnit.KILOGRAM


def test_price_basis_rejects_wrong_measurement_family() -> None:
    """A per-weight price cannot carry a volume denominator."""
    with pytest.raises(ValidationError, match="weight price_basis_unit"):
        make_offer(
            "Ground Beef",
            price_basis=PriceBasis.PER_WEIGHT,
            price_basis_unit=MeasurementUnit.LITRE,
        )
