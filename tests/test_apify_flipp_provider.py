"""Offline contract tests for the Apify-backed grocery search provider."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.providers.apify_flipp import ApifyFlippProvider
from app.providers.base import (
    GrocerySearchProvider,
    SearchProviderExecutionError,
    SearchProviderTimeoutError,
    UnsupportedSearchIntentError,
)
from app.schemas.flyer_search import RawFlyerSearchRequest, RawFlyerSearchResult
from app.schemas.product_offer import ProductOffer
from app.schemas.search_policy import SearchRequestPlan
from app.schemas.search_provider import GrocerySearchIntent, GrocerySearchRequest
from app.schemas.shopping import (
    ConstraintRequirement,
    ShoppingConstraint,
    ShoppingItem,
)
from app.services.apify_flyer_transformer import (
    ApifyFlyerTransformationError,
    transform_apify_flyer_result,
)
from app.services.raw_flyer_search import (
    FLIPP_FLYER_TOOL_NAME,
    RawFlyerSearchTimeoutError,
)
from app.services.search_request_policy import (
    ExternalActorBudgetExceededError,
    ExternalActorCallBudget,
)

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "apify" / "pipeline_valid_response.json"
)
RETRIEVED_AT = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def make_request(
    *,
    intent: GrocerySearchIntent = GrocerySearchIntent.FLYER_DEALS,
    max_results: int = 10,
    store: str | None = "Example Grocer",
) -> GrocerySearchRequest:
    """Create one shared request containing meaningful product qualifiers."""
    return GrocerySearchRequest(
        item=ShoppingItem(
            name="Milk",
            constraints=(
                ShoppingConstraint(value="2%"),
                ShoppingConstraint(
                    value="organic",
                    requirement=ConstraintRequirement.OPTIONAL,
                ),
            ),
        ),
        postal_code="m 5 v 3 a 8",
        intent=intent,
        store=store,
        max_results=max_results,
    )


def make_budget(
    request: GrocerySearchRequest,
    *,
    max_calls: int = 1,
) -> ExternalActorCallBudget:
    """Build the application-owned paid-call budget required by the provider."""
    plan = SearchRequestPlan(
        items=(request.item,),
        max_external_actor_calls=max_calls,
        max_concurrency=1,
    )
    return ExternalActorCallBudget(plan)


def load_validated_fixture_offers() -> list[ProductOffer]:
    """Transform a saved MCP fixture without discovery, network, or Actor usage."""
    request = RawFlyerSearchRequest(
        query="cantaloupe",
        postal_code="M5V 3A8",
    )
    raw_result = RawFlyerSearchResult(
        tool_name=FLIPP_FLYER_TOOL_NAME,
        request=request,
        raw_response=json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
    )
    return transform_apify_flyer_result(raw_result)


@pytest.mark.anyio
async def test_provider_translates_shared_request_and_returns_shared_result() -> None:
    """Apify-specific inputs and fixture output should remain behind the provider."""
    request = make_request()
    offers = load_validated_fixture_offers()
    offer_search = AsyncMock(return_value=offers)
    provider = ApifyFlippProvider(
        call_budget=make_budget(request),
        timeout_seconds=15,
        offer_search=offer_search,
        clock=lambda: RETRIEVED_AT,
    )

    result = await provider.search(request)

    assert isinstance(provider, GrocerySearchProvider)
    offer_search.assert_awaited_once()
    call = offer_search.await_args
    assert call is not None
    raw_request = call.args[0]
    assert type(raw_request) is RawFlyerSearchRequest
    # Required qualifiers narrow the Actor query. Optional qualifiers remain on
    # result.request for later matching without excluding useful alternatives.
    assert raw_request.query == "2% Milk"
    assert result.request.item.constraints[1].value == "organic"
    assert raw_request.postal_code == "M5V 3A8"
    assert raw_request.merchant_name == "Example Grocer"
    assert raw_request.to_tool_input()["merchantName"] == "Example Grocer"
    # The shared cap cannot raise the existing cost-conscious Actor limit.
    assert raw_request.max_items == 5
    assert call.kwargs == {"timeout_seconds": 15}

    assert result.request is request
    assert result.offers == tuple(offers)
    assert result.provenance.provider_name == "apify_flipp"
    assert result.provenance.retrieved_at == RETRIEVED_AT
    assert result.offers[0].source == "https://example.invalid/flyer/cantaloupe"
    assert "raw_response" not in type(result).model_fields


@pytest.mark.anyio
async def test_provider_returns_a_valid_empty_result() -> None:
    """A successful flyer search with no offers should remain distinct from failure."""
    request = make_request(max_results=1, store=None)
    offer_search = AsyncMock(return_value=[])
    provider = ApifyFlippProvider(
        call_budget=make_budget(request),
        offer_search=offer_search,
        clock=lambda: RETRIEVED_AT,
    )

    result = await provider.search(request)

    assert result.offers == ()
    call = offer_search.await_args
    assert call is not None
    raw_request = call.args[0]
    assert raw_request.max_items == 1
    assert raw_request.merchant_name is None
    assert "merchantName" not in raw_request.to_tool_input()


@pytest.mark.anyio
async def test_provider_rejects_regular_price_intent_without_spending_budget() -> None:
    """Unsupported intent must fail before entering the paid Actor boundary."""
    request = make_request(intent=GrocerySearchIntent.REGULAR_PRICES)
    budget = make_budget(request)
    offer_search = AsyncMock()
    provider = ApifyFlippProvider(
        call_budget=budget,
        offer_search=offer_search,
    )

    with pytest.raises(UnsupportedSearchIntentError, match="flyer-deal"):
        await provider.search(request)

    offer_search.assert_not_awaited()
    assert budget.remaining_calls == 1


@pytest.mark.anyio
async def test_provider_rejects_an_oversized_actor_query_before_spending_budget() -> (
    None
):
    """A valid shared item may still exceed the Actor's stricter query boundary."""
    request = GrocerySearchRequest(
        item=ShoppingItem(name="m" * 101),
        postal_code="M5V 3A8",
    )
    budget = make_budget(request)
    offer_search = AsyncMock()
    provider = ApifyFlippProvider(
        call_budget=budget,
        offer_search=offer_search,
    )

    with pytest.raises(SearchProviderExecutionError, match="represented safely"):
        await provider.search(request)

    offer_search.assert_not_awaited()
    assert budget.remaining_calls == 1


@pytest.mark.anyio
async def test_provider_consumes_budget_and_never_retries_failed_pipeline() -> None:
    """A malformed response should consume one slot and surface a safe error once."""
    request = make_request()
    budget = make_budget(request)
    offer_search = AsyncMock(
        side_effect=ApifyFlyerTransformationError(
            "provider response contained secret-looking diagnostic text"
        )
    )
    provider = ApifyFlippProvider(
        call_budget=budget,
        offer_search=offer_search,
    )

    with pytest.raises(
        SearchProviderExecutionError,
        match="could not produce validated offers",
    ) as error:
        await provider.search(request)

    assert "secret-looking" not in str(error.value)
    assert offer_search.await_count == 1
    assert budget.remaining_calls == 0


@pytest.mark.anyio
async def test_provider_preserves_timeout_as_a_distinct_safe_error() -> None:
    """Callers should recognize a timeout without receiving transport details."""
    request = make_request()
    offer_search = AsyncMock(
        side_effect=RawFlyerSearchTimeoutError("Authorization: Bearer hidden")
    )
    provider = ApifyFlippProvider(
        call_budget=make_budget(request),
        offer_search=offer_search,
    )

    with pytest.raises(SearchProviderTimeoutError, match="not retried") as error:
        await provider.search(request)

    assert "Bearer" not in str(error.value)
    assert offer_search.await_count == 1


@pytest.mark.anyio
async def test_provider_cannot_exceed_the_request_actor_call_budget() -> None:
    """A second search cannot invoke the integration after its budget is exhausted."""
    request = make_request()
    offer_search = AsyncMock(return_value=[])
    provider = ApifyFlippProvider(
        call_budget=make_budget(request),
        offer_search=offer_search,
        clock=lambda: RETRIEVED_AT,
    )

    await provider.search(request)
    with pytest.raises(ExternalActorBudgetExceededError, match="exhausted"):
        await provider.search(request)

    assert offer_search.await_count == 1
