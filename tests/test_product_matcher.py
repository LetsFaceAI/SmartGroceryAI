"""Focused tests for conservative deterministic product-name matching."""

import pytest
from pydantic import ValidationError

from app.schemas.normalized_product import NormalizedProduct
from app.schemas.product_match import (
    ProductMatchDecision,
    ProductMatchResult,
    ProductMatchType,
)
from app.schemas.product_offer import ProductOffer
from app.services.product_matcher import ProductMatchingError, match_product
from app.services.product_normalization import normalize_product_offer


def make_product(product_name: str) -> NormalizedProduct:
    """Create a comparison-ready fixture through the production normalizer."""
    offer = ProductOffer(
        product_name=product_name,
        store="Example Grocer",
        price="3.99",
        source=f"fixture:{product_name}",
    )
    return normalize_product_offer(offer)


def test_exact_normalized_name_match() -> None:
    """Casing, whitespace, and formatting noise should not prevent equality."""
    result = match_product("  ORGANIC®   Milk ", make_product("Organic Milk"))

    assert type(result) is ProductMatchResult
    assert result.matched is True
    assert result.match_type is ProductMatchType.EXACT
    assert result.confidence == 1.0
    assert result.normalized_request_name == "organic milk"


@pytest.mark.parametrize(
    ("requested_name", "flyer_name"),
    [
        ("milk", "Organic 2% Milk"),
        ("ground beef", "Lean Ground Beef"),
        ("large eggs", "Large Eggs 12 Count"),
    ],
)
def test_complete_token_sequence_containment_matches(
    requested_name: str,
    flyer_name: str,
) -> None:
    """Extra product qualifiers may surround the complete requested phrase."""
    result = match_product(requested_name, make_product(flyer_name))

    assert result.matched is True
    assert result.match_type is ProductMatchType.CONTAINMENT
    assert result.confidence == 0.85


def test_strong_reordered_token_overlap_is_an_uncertain_candidate() -> None:
    """Reordered words stay visible without being trusted for automatic ranking."""
    result = match_product(
        "tomato basil pasta sauce",
        make_product("Basil Tomato Sauce Pasta"),
    )

    assert result.matched is False
    assert result.candidate is True
    assert result.safe_for_ranking is False
    assert result.decision is ProductMatchDecision.UNCERTAIN
    assert result.match_type is ProductMatchType.TOKEN_OVERLAP
    assert result.confidence == 0.70


@pytest.mark.parametrize(
    ("requested_name", "flyer_name"),
    [
        ("milk", "ground beef"),
        ("apple juice", "orange juice"),
        ("ham", "hamburger buns"),
        ("organic apple juice", "organic orange juice"),
        ("large brown eggs", "large white eggs"),
        ("organic milk", "milk"),
        ("2% milk", "milk"),
    ],
)
def test_obvious_or_weak_matches_are_rejected(
    requested_name: str,
    flyer_name: str,
) -> None:
    """Shared generic words and raw substrings must not create false positives."""
    result = match_product(requested_name, make_product(flyer_name))

    assert result.matched is False
    assert result.match_type is ProductMatchType.NONE
    assert result.confidence == 0.0


@pytest.mark.parametrize("requested_name", ["", "   ", "® ™ ©", "---"])
def test_empty_normalized_request_fails_clearly(requested_name: str) -> None:
    """A request without comparable words cannot produce a safe decision."""
    with pytest.raises(ProductMatchingError, match="at least one letter or number"):
        match_product(requested_name, make_product("Milk"))


def test_match_result_retains_normalized_product() -> None:
    """Downstream comparison can use the matched product without another lookup."""
    product = make_product("Whole Milk")

    result = match_product("whole milk", product)

    assert result.product is product


@pytest.mark.parametrize(
    "contradictory_fields",
    [
        {"matched": False, "match_type": ProductMatchType.EXACT},
        {"matched": True, "match_type": ProductMatchType.NONE},
        {"matched": False, "match_type": ProductMatchType.NONE, "confidence": 0.5},
    ],
)
def test_match_result_rejects_contradictory_decisions(
    contradictory_fields: dict[str, object],
) -> None:
    """Consumers should never receive conflicting flags, types, or confidence."""
    result_data = match_product("milk", make_product("milk")).model_dump()
    result_data.update(contradictory_fields)

    with pytest.raises(ValidationError):
        ProductMatchResult.model_validate(result_data)
