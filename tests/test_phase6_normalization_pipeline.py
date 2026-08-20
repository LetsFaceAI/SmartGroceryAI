"""Offline integration tests for the complete deterministic Phase 6 boundary.

These tests start with Pydantic's external ``ProductOffer`` contract, then execute
real normalization, package conversion, and matching code together. No MCP, LLM,
embedding model, vector store, or other external dependency is involved.
"""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.normalized_product import CanonicalUnit, NormalizedProduct
from app.schemas.product_match import ProductMatchResult, ProductMatchType
from app.schemas.product_offer import MeasurementUnit, ProductOffer, PromotionStatus
from app.services.product_matcher import match_product
from app.services.product_normalization import (
    ProductNormalizationError,
    normalize_product_offer,
    parse_package_size,
)
from app.services.product_offer_mapper import map_product_offer


def test_complete_offer_normalizes_and_matches_end_to_end() -> None:
    """A complete offer should become a canonical, matchable product."""
    offer = ProductOffer(
        product_name="  Organic® 2%   Milk ",
        brand="Example Dairy™",
        store="Walmart Canada",
        package_size="1.5",
        unit=MeasurementUnit.LITRE,
        price="4.49",
        regular_price="5.49",
        promotion_status=PromotionStatus.SALE,
        source="fixture:milk-001",
    )

    product = normalize_product_offer(offer)
    result = match_product("2% milk", product)

    assert type(product) is NormalizedProduct
    assert product.normalized_name == "organic 2% milk"
    assert product.brand == "example dairy"
    assert product.store == "walmart"
    assert product.package_quantity == 1
    assert product.package_size == Decimal("1500.0")
    assert product.total_package_size == Decimal("1500.0")
    assert product.unit is CanonicalUnit.MILLILITRE
    assert product.original_offer is offer
    assert type(result) is ProductMatchResult
    assert result.matched is True
    assert result.match_type is ProductMatchType.CONTAINMENT


def test_partial_offer_preserves_unknown_values_and_still_matches() -> None:
    """Missing optional flyer fields should remain absent across the full flow."""
    offer = ProductOffer(
        product_name="Large Eggs",
        store="Local Market",
        price="3.99",
        package_size="12",
        source="fixture:eggs-001",
    )

    product = normalize_product_offer(offer)
    result = match_product("large eggs", product)

    assert product.brand is None
    assert product.package_quantity is None
    assert product.package_size is None
    assert product.total_package_size is None
    assert product.unit is None
    assert product.regular_price is None
    assert product.promotion_status is PromotionStatus.UNKNOWN
    # A size without a unit stays available on the source offer, but canonical
    # package fields remain empty rather than guessing "count" or "pack".
    assert offer.package_size == Decimal("12")
    assert result.match_type is ProductMatchType.EXACT


def test_invalid_offer_fails_before_normalization_or_matching() -> None:
    """Malformed external pricing must stop at the ProductOffer boundary."""
    with pytest.raises(ValidationError, match="price"):
        ProductOffer(
            product_name="Milk",
            store="Example Grocer",
            price="0",
            source="fixture:invalid-price",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("product_name", "®™", "product name"),
        ("store", "®™", "store name"),
    ],
)
def test_formatting_only_identity_text_fails_normalization_clearly(
    field: str,
    value: str,
    message: str,
) -> None:
    """A syntactically valid offer must not produce an empty canonical identity."""
    offer_data: dict[str, object] = {
        "product_name": "Milk",
        "store": "Example Grocer",
        "price": "3.99",
        "source": "fixture:formatting-only",
    }
    offer_data[field] = value
    offer = ProductOffer.model_validate(offer_data)

    with pytest.raises(ProductNormalizationError, match=message):
        normalize_product_offer(offer)


def test_non_matching_offer_returns_explicit_none_result() -> None:
    """An unrelated product should remain available but receive no match."""
    offer = ProductOffer(
        product_name="Orange Juice",
        store="Example Grocer",
        price="4.99",
        source="fixture:orange-juice",
    )

    product = normalize_product_offer(offer)
    result = match_product("ground beef", product)

    assert result.matched is False
    assert result.match_type is ProductMatchType.NONE
    assert result.confidence == 0.0
    assert result.product is product


def test_multipack_parser_is_canonical_but_rejects_unsupported_units() -> None:
    """Package parsing should calculate exact totals without guessing new units."""
    package = parse_package_size("2 x 500 g")

    assert package.package_quantity == 2
    assert package.package_size == Decimal("500")
    assert package.total_package_size == Decimal("1000")
    assert package.unit is CanonicalUnit.GRAM

    with pytest.raises(ProductNormalizationError, match="Unsupported measurement unit"):
        parse_package_size("3 bunches")


def test_mapped_multipack_reaches_normalization_and_matching() -> None:
    """The validated offer boundary must retain an embedded package multiplier."""
    offer = map_product_offer(
        {
            "product": "Lean Ground Beef",
            "store": "Example Grocer",
            "price": "8.99",
            "packageSize": "2 x 500 g",
            "source": "fixture:multipack-beef",
        }
    )

    product = normalize_product_offer(offer)
    result = match_product("ground beef", product)

    assert offer.package_quantity == 2
    assert offer.package_size == Decimal("500")
    assert offer.unit is MeasurementUnit.GRAM
    assert product.package_quantity == 2
    assert product.package_size == Decimal("500")
    assert product.total_package_size == Decimal("1000")
    assert product.unit is CanonicalUnit.GRAM
    assert result.matched is True
    assert result.match_type is ProductMatchType.CONTAINMENT
