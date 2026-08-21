"""Full offline Phase 7.1 pipeline coverage using a saved Apify response."""

import json
from datetime import date
from pathlib import Path

from app.schemas.flyer_search import RawFlyerSearchRequest, RawFlyerSearchResult
from app.schemas.price_comparison import OfferValidityStatus, PriceComparisonStatus
from app.schemas.product_match import ProductMatchDecision
from app.schemas.product_offer import PriceBasis
from app.schemas.shopping import (
    ConstraintRequirement,
    ShoppingConstraint,
    ShoppingItem,
)
from app.services.apify_flyer_transformer import transform_apify_flyer_result
from app.services.price_comparison import calculate_unit_price
from app.services.product_matcher import build_product_match_request, match_product
from app.services.product_normalization import normalize_product_offer

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "apify" / "flipp_flyer_items.json"
AS_OF = date(2026, 8, 20)


def test_saved_mcp_fixture_flows_through_phase7_1_boundaries() -> None:
    """Validated fixture rows should be compared only when semantics are proven."""
    raw_result = RawFlyerSearchResult(
        tool_name="crawlerbros--flipp-grocery-deals-scraper",
        request=RawFlyerSearchRequest(
            query="cantaloupe",
            postal_code="M5V 3A8",
            max_items=2,
        ),
        raw_response=json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
    )

    offers = transform_apify_flyer_result(raw_result)
    products = [normalize_product_offer(offer) for offer in offers]

    cantaloupe_request = build_product_match_request(
        ShoppingItem(
            name="cantaloupe",
            constraints=(
                ShoppingConstraint(
                    value="organic",
                    requirement=ConstraintRequirement.OPTIONAL,
                ),
            ),
        )
    )
    cantaloupe_match = match_product(cantaloupe_request, products[0])
    cantaloupe_comparison = calculate_unit_price(products[0], as_of=AS_OF)

    assert cantaloupe_match.decision is ProductMatchDecision.SAFE
    assert cantaloupe_match.unmet_optional_constraints == ("organic",)
    assert offers[0].price_basis is PriceBasis.EACH
    assert cantaloupe_comparison.status is PriceComparisonStatus.COMPARABLE

    # The second live-style row omits dates, package data, and priceQualifier.
    # Those omissions remain explicit instead of being guessed from the product.
    eggs_match = match_product("large eggs", products[1])
    eggs_comparison = calculate_unit_price(products[1], as_of=AS_OF)

    assert eggs_match.decision is ProductMatchDecision.SAFE
    assert offers[1].price_basis is PriceBasis.UNKNOWN
    assert eggs_comparison.validity_status is OfferValidityStatus.UNKNOWN
    assert eggs_comparison.status is PriceComparisonStatus.INELIGIBLE_VALIDITY
    assert eggs_comparison.unit_price is None
