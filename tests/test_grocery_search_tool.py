"""Offline tests for the LangChain grocery-search tool."""

from datetime import date

import pytest

from app.schemas.price_comparison import (
    CheapestOfferSelection,
    CheapestOfferStatus,
)
from app.schemas.search_provider import GrocerySearchRequest
from app.services.grocery_search import GrocerySearchService
from app.tools.grocery_search import create_find_flyer_deals_tool


class FakeGrocerySearchService(GrocerySearchService):
    """Record tool requests without contacting a provider."""

    def __init__(self, selection: CheapestOfferSelection) -> None:
        self._selection = selection
        self.calls: list[tuple[GrocerySearchRequest, date]] = []

    async def compare_offers(
        self,
        request: GrocerySearchRequest,
        *,
        as_of: date,
    ) -> CheapestOfferSelection:
        self.calls.append((request, as_of))
        return self._selection


@pytest.mark.anyio
async def test_tool_exposes_schema_and_invokes_application_service() -> None:
    """LangChain should validate inputs and invoke only the narrow service."""
    selection = CheapestOfferSelection(
        requested_item_name="milk",
        status=CheapestOfferStatus.NO_COMPARABLE_OFFERS,
        reason="No trusted active offers were available.",
        comparisons=(),
    )
    service = FakeGrocerySearchService(selection)
    grocery_tool = create_find_flyer_deals_tool(service)

    result = await grocery_tool.ainvoke(
        {
            "item_name": "milk",
            "postal_code": "m5v3a8",
            "store": "No Frills",
        }
    )

    assert grocery_tool.name == "find_flyer_deals"
    assert set(grocery_tool.args) == {"item_name", "postal_code", "store"}

    assert len(service.calls) == 1
    request, as_of = service.calls[0]
    assert request.item.name == "milk"
    assert request.postal_code == "M5V 3A8"
    assert request.store == "No Frills"
    assert isinstance(as_of, date)

    assert result["status"] == "no_comparable_offers"
    assert result["requested_item_name"] == "milk"