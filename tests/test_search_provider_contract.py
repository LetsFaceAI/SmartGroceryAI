"""Unit tests for the provider-neutral grocery search contract."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.providers.base import GrocerySearchProvider
from app.schemas.product_offer import ProductOffer
from app.schemas.search_provider import (
    GrocerySearchIntent,
    GrocerySearchRequest,
    GrocerySearchResult,
    SearchProvenance,
)
from app.schemas.shopping import ShoppingConstraint, ShoppingItem


def make_request(**overrides: object) -> GrocerySearchRequest:
    """Build a representative shared request with optional field overrides."""
    values: dict[str, object] = {
        "item": ShoppingItem(
            name="Milk",
            constraints=(ShoppingConstraint(value="2%"),),
        ),
        "postal_code": "m 5 v 3 a 8",
        "max_results": 2,
    }
    values.update(overrides)
    return GrocerySearchRequest.model_validate(values)


def make_offer(*, source: str = "https://example.test/offer/1") -> ProductOffer:
    """Build one validated offer used by shared-result tests."""
    return ProductOffer(
        product_name="2% Milk",
        store="Example Market",
        price=Decimal("4.99"),
        source=source,
    )


def test_request_preserves_item_constraints_and_normalizes_location() -> None:
    """Provider translation must receive qualifiers and one canonical location."""
    request = make_request()

    assert request.item.name == "Milk"
    assert request.item.constraints[0].value == "2%"
    assert request.postal_code == "M5V 3A8"
    assert request.intent is GrocerySearchIntent.FLYER_DEALS


def test_request_supports_regular_price_and_store_specific_search() -> None:
    """Store scoping should compose with price intent instead of replacing it."""
    request = make_request(
        intent=GrocerySearchIntent.REGULAR_PRICES,
        store="  Metro  ",
    )

    assert request.intent is GrocerySearchIntent.REGULAR_PRICES
    assert request.store == "Metro"


@pytest.mark.parametrize("postal_code", ["", "Toronto", "1234", "M5V-3A8"])
def test_request_rejects_invalid_search_locations(postal_code: str) -> None:
    """Invalid locations should fail before any provider can perform network I/O."""
    with pytest.raises(ValidationError, match="postal_code"):
        make_request(postal_code=postal_code)


def test_result_preserves_provider_and_record_level_provenance() -> None:
    """Shared results should identify both their provider and each offer source."""
    offer = make_offer()
    result = GrocerySearchResult(
        request=make_request(),
        offers=(offer,),
        provenance=SearchProvenance(
            provider_name="apify",
            retrieved_at=datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
            provider_request_id="actor-run-123",
        ),
    )

    assert result.provenance.provider_name == "apify"
    assert result.provenance.provider_request_id == "actor-run-123"
    assert result.offers[0].source == "https://example.test/offer/1"


def test_result_allows_a_valid_search_with_no_matching_offers() -> None:
    """No matches is a valid provider outcome and should not resemble a failure."""
    result = GrocerySearchResult(
        request=make_request(),
        provenance=SearchProvenance(
            provider_name="apify",
            retrieved_at=datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
        ),
    )

    assert result.offers == ()


def test_result_rejects_provider_raw_payloads_and_excess_results() -> None:
    """Raw provider data stays outside the shared model and result caps are binding."""
    request = make_request(max_results=1)
    provenance = SearchProvenance(
        provider_name="tavily",
        retrieved_at=datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
    )

    with pytest.raises(ValidationError, match="raw_payload"):
        GrocerySearchResult.model_validate(
            {
                "request": request,
                "offers": [make_offer()],
                "provenance": provenance,
                "raw_payload": {"provider_specific": True},
            }
        )

    with pytest.raises(ValidationError, match="max_results"):
        GrocerySearchResult(
            request=request,
            offers=(make_offer(source="one"), make_offer(source="two")),
            provenance=provenance,
        )


def test_provenance_requires_an_aware_timestamp() -> None:
    """A timezone-free retrieval time would make cross-provider ordering ambiguous."""
    with pytest.raises(ValidationError, match="retrieved_at"):
        SearchProvenance(
            provider_name="apify",
            retrieved_at=datetime(2026, 8, 20, 12, 0),
        )


@pytest.mark.anyio
async def test_protocol_accepts_an_independent_provider_implementation() -> None:
    """Callers can use a provider without importing its SDK or transport types."""

    class FakeProvider:
        @property
        def provider_name(self) -> str:
            return "fixture"

        async def search(self, request: GrocerySearchRequest) -> GrocerySearchResult:
            return GrocerySearchResult(
                request=request,
                offers=(make_offer(),),
                provenance=SearchProvenance(
                    provider_name=self.provider_name,
                    retrieved_at=datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
                ),
            )

    provider: GrocerySearchProvider = FakeProvider()
    result = await provider.search(make_request())

    assert isinstance(provider, GrocerySearchProvider)
    assert result.provenance.provider_name == "fixture"
    assert type(result.offers[0]) is ProductOffer
