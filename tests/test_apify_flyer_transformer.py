"""Offline contract tests for converting saved Flipp data to ProductOffer."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.schemas.flyer_search import RawFlyerSearchRequest, RawFlyerSearchResult
from app.schemas.product_offer import ProductOffer, PromotionStatus
from app.services.apify_flyer_transformer import (
    ApifyFlyerTransformationError,
    map_apify_flyer_record,
    search_product_offers,
    transform_apify_flyer_result,
)

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "apify"


def load_fixture(filename: str) -> dict[str, Any]:
    """Load a committed Actor-shaped response without contacting Apify."""
    fixture_path = FIXTURE_DIRECTORY / filename
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def make_result(raw_response: object) -> RawFlyerSearchResult:
    """Wrap fixture data exactly as the raw search service would return it."""
    return RawFlyerSearchResult(
        tool_name="crawlerbros--flipp-grocery-deals-scraper",
        request=RawFlyerSearchRequest(
            query="cantaloupe",
            postal_code="M5V 3A8",
            max_items=2,
        ),
        raw_response=raw_response,
    )


def test_transform_complete_saved_flyer_result() -> None:
    """A complete Actor row should map every supported reliable field."""
    offers = transform_apify_flyer_result(
        make_result(load_fixture("flipp_flyer_items.json"))
    )

    offer = offers[0]
    assert type(offer) is ProductOffer
    assert offer.product_name == "Whole Cantaloupe"
    assert offer.brand == "Fresh Market"
    assert offer.store == "Example Grocer"
    assert offer.price == Decimal("2.99")
    assert offer.regular_price == Decimal("3.99")
    assert offer.currency == "CAD"
    assert offer.promotion_status is PromotionStatus.SALE
    assert offer.valid_from.isoformat() == "2026-08-20"
    assert offer.valid_until.isoformat() == "2026-08-26"
    assert offer.source == "https://example.invalid/flyer/cantaloupe"


def test_transform_partial_saved_flyer_result_uses_safe_defaults() -> None:
    """Missing optional Actor fields should remain absent, not be invented."""
    offers = transform_apify_flyer_result(
        make_result(load_fixture("flipp_flyer_items.json"))
    )

    offer = offers[1]
    assert offer.brand is None
    assert offer.regular_price is None
    assert offer.valid_from is None
    assert offer.valid_until is None
    assert offer.promotion_status is PromotionStatus.UNKNOWN
    assert offer.source == "apify:flipp:flipp-eggs-002"


def test_transform_rejects_malformed_saved_flyer_result_clearly() -> None:
    """Bad required fields should identify the failing dataset row."""
    with pytest.raises(
        ApifyFlyerTransformationError,
        match=r"record at index 0.*Invalid Flipp flyer record",
    ):
        transform_apify_flyer_result(
            make_result(load_fixture("flipp_flyer_malformed.json"))
        )


def test_transform_reads_langchain_structured_artifact() -> None:
    """Machine-readable MCP artifacts should work without parsing display text."""
    fixture = load_fixture("flipp_flyer_items.json")
    response = ToolMessage(
        content="SUCCEEDED. Preview contains structured items.",
        artifact={"structured_content": fixture},
        tool_call_id="fixture-call",
        name="crawlerbros--flipp-grocery-deals-scraper",
    )

    offers = transform_apify_flyer_result(make_result(response))

    assert [offer.product_name for offer in offers] == [
        "Whole Cantaloupe",
        "Large Eggs",
    ]


def test_map_record_rejects_unknown_currency_location() -> None:
    """Currency must not silently default when neither record nor location proves it."""
    record = load_fixture("flipp_flyer_items.json")["items"][0]

    with pytest.raises(ApifyFlyerTransformationError, match="currency"):
        map_apify_flyer_record(record, postal_code="UNKNOWN")


def test_transform_rejects_unstructured_actor_summary() -> None:
    """Human-readable run summaries are not a safe substitute for dataset fields."""
    with pytest.raises(
        ApifyFlyerTransformationError,
        match="did not contain structured",
    ):
        transform_apify_flyer_result(
            make_result([{"type": "text", "text": "SUCCEEDED. 1 item; 18 fields."}])
        )


@pytest.mark.anyio
async def test_search_product_offers_reuses_raw_service_without_network() -> None:
    """The composed service should call the raw boundary once and map its result."""
    raw_result = make_result(load_fixture("flipp_flyer_items.json"))
    client = Mock(spec=MultiServerMCPClient)

    with patch(
        "app.services.apify_flyer_transformer.search_raw_flyer_offers",
        new=AsyncMock(return_value=raw_result),
    ) as raw_search:
        offers = await search_product_offers(
            raw_result.request,
            client=client,
            timeout_seconds=1,
        )

    raw_search.assert_awaited_once_with(
        raw_result.request,
        client=client,
        timeout_seconds=1,
    )
    assert len(offers) == 2
