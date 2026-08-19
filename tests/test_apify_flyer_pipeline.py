"""Integration-style tests for the offline Apify flyer data boundary.

Only the raw MCP execution service is mocked. Response extraction, Flipp field
transformation, the reusable mapper, and ProductOffer validation run together so
these tests catch contract mismatches without spending Apify credit.
"""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.schemas.flyer_search import RawFlyerSearchRequest, RawFlyerSearchResult
from app.schemas.product_offer import ProductOffer, PromotionStatus
from app.services.apify_flyer_transformer import (
    ApifyFlyerTransformationError,
    search_product_offers,
)

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "apify"
ACTOR_TOOL_NAME = "crawlerbros--flipp-grocery-deals-scraper"


def load_fixture(filename: str) -> dict[str, Any]:
    """Read a committed MCP-shaped response; this helper performs no network I/O."""
    fixture_path = FIXTURE_DIRECTORY / filename
    return json.loads(fixture_path.read_text(encoding="utf-8"))


async def run_offline_pipeline(
    raw_response: object,
    *,
    max_items: int = 1,
) -> list[ProductOffer]:
    """Mock the paid boundary while exercising every downstream production layer."""
    request = RawFlyerSearchRequest(
        query="fixture query",
        postal_code="M5V 3A8",
        max_items=max_items,
    )
    raw_result = RawFlyerSearchResult(
        tool_name=ACTOR_TOOL_NAME,
        request=request,
        raw_response=raw_response,
    )
    client = Mock(spec=MultiServerMCPClient)

    with patch(
        "app.services.apify_flyer_transformer.search_raw_flyer_offers",
        new=AsyncMock(return_value=raw_result),
    ) as raw_search:
        offers = await search_product_offers(
            request,
            client=client,
            timeout_seconds=1,
        )

    # This assertion guards against a future refactor accidentally invoking the
    # cost-sensitive Actor more than once per application request.
    raw_search.assert_awaited_once_with(
        request,
        client=client,
        timeout_seconds=1,
    )
    return offers


@pytest.mark.anyio
async def test_pipeline_maps_valid_mcp_response_to_product_offer() -> None:
    """A complete structured MCP preview should cross the full data boundary."""
    offers = await run_offline_pipeline(load_fixture("pipeline_valid_response.json"))

    assert len(offers) == 1
    offer = offers[0]
    assert type(offer) is ProductOffer
    assert offer.product_name == "Whole Cantaloupe"
    assert offer.store == "Example Grocer"
    assert offer.price == Decimal("2.99")
    assert offer.regular_price == Decimal("3.99")
    assert offer.promotion_status is PromotionStatus.SALE
    assert offer.valid_from.isoformat() == "2026-08-20"
    assert offer.valid_until.isoformat() == "2026-08-26"


@pytest.mark.anyio
async def test_pipeline_accepts_missing_optional_flyer_fields() -> None:
    """A partial provider row should preserve absence through schema defaults."""
    offers = await run_offline_pipeline(load_fixture("pipeline_partial_response.json"))

    offer = offers[0]
    assert type(offer) is ProductOffer
    assert offer.brand is None
    assert offer.regular_price is None
    assert offer.valid_from is None
    assert offer.valid_until is None
    assert offer.promotion_status is PromotionStatus.UNKNOWN
    assert offer.source == "apify:flipp:pipeline-eggs-002"


@pytest.mark.anyio
async def test_pipeline_uses_normalized_canadian_postal_code() -> None:
    """The canonical schema value should reach both execution and transformation."""
    request = RawFlyerSearchRequest(
        query="fixture query",
        postal_code="m 5 v 3 a 8",
    )
    raw_result = RawFlyerSearchResult(
        tool_name=ACTOR_TOOL_NAME,
        request=request,
        raw_response=load_fixture("pipeline_partial_response.json"),
    )
    client = Mock(spec=MultiServerMCPClient)

    with patch(
        "app.services.apify_flyer_transformer.search_raw_flyer_offers",
        new=AsyncMock(return_value=raw_result),
    ) as raw_search:
        offers = await search_product_offers(
            request,
            client=client,
            timeout_seconds=1,
        )

    assert request.postal_code == "M5V 3A8"
    raw_search.assert_awaited_once_with(
        request,
        client=client,
        timeout_seconds=1,
    )
    # The fixture has no explicit currency, so CAD proves the transformer used
    # the normalized Canadian location rather than an unvalidated raw string.
    assert offers[0].currency == "CAD"


@pytest.mark.anyio
async def test_pipeline_rejects_missing_required_flyer_fields() -> None:
    """Missing store and provenance must fail instead of producing a partial offer."""
    with pytest.raises(
        ApifyFlyerTransformationError,
        match=r"record at index 0.*source, store",
    ):
        await run_offline_pipeline(load_fixture("pipeline_malformed_response.json"))


@pytest.mark.anyio
async def test_pipeline_rejects_invalid_price() -> None:
    """A non-positive provider price must be rejected by ProductOffer validation."""
    response = load_fixture("pipeline_invalid_responses.json")["invalid_price"]

    with pytest.raises(
        ApifyFlyerTransformationError,
        match=r"record at index 0.*price",
    ):
        await run_offline_pipeline(response)


@pytest.mark.anyio
async def test_pipeline_rejects_invalid_date_range() -> None:
    """An end date before the start date must surface the deterministic rule."""
    response = load_fixture("pipeline_invalid_responses.json")["invalid_dates"]

    with pytest.raises(
        ApifyFlyerTransformationError,
        match=r"record at index 0.*valid_until cannot be earlier than valid_from",
    ):
        await run_offline_pipeline(response)


@pytest.mark.anyio
async def test_pipeline_returns_multiple_validated_offers() -> None:
    """Every row in a multi-item MCP preview should become a ProductOffer."""
    offers = await run_offline_pipeline(
        load_fixture("pipeline_multiple_response.json"),
        max_items=2,
    )

    assert len(offers) == 2
    assert all(type(offer) is ProductOffer for offer in offers)
    assert [offer.product_name for offer in offers] == [
        "Gala Apples",
        "Whole Wheat Bread",
    ]
    assert offers[1].promotion_status is PromotionStatus.SALE
