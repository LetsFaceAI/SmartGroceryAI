"""Offline tests for the LangChain grocery-search tool."""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from app.schemas.offer_candidates import (
    FlyerOfferCandidate,
    FlyerOfferCandidateSet,
)
from app.schemas.product_offer import PriceBasis, ProductOffer
from app.schemas.search_provider import GrocerySearchRequest
from app.schemas.shopping import (
    ConstraintRequirement,
    ShoppingConstraint,
    ShoppingItem,
)
from app.services.grocery_search import GrocerySearchService
from app.tools.grocery_search import create_find_flyer_deals_tool

AS_OF = date(2026, 8, 20)


class FakeGrocerySearchService(GrocerySearchService):
    """Record tool requests without contacting a provider."""

    def __init__(self, candidates: FlyerOfferCandidateSet) -> None:
        self._candidates = candidates
        self.calls: list[tuple[GrocerySearchRequest, date]] = []

    async def search_offer_candidates(
        self,
        request: GrocerySearchRequest,
        *,
        as_of: date,
    ) -> FlyerOfferCandidateSet:
        self.calls.append((request, as_of))
        return self._candidates


def make_candidates() -> FlyerOfferCandidateSet:
    """Create one unranked candidate set for tool-boundary tests."""
    offer = ProductOffer(
        product_name="Crown Broccoli",
        store="Nations Fresh Foods",
        price=Decimal("1.29"),
        price_basis=PriceBasis.UNKNOWN,
        valid_from=date(2026, 8, 14),
        valid_until=date(2026, 8, 20),
        source="fixture:nations",
    )
    return FlyerOfferCandidateSet(
        requested_item_name="broccoli",
        postal_code="L5B 0G7",
        as_of=AS_OF,
        provider_name="fixture",
        retrieved_at=datetime(2026, 8, 20, tzinfo=UTC),
        offers=(
            FlyerOfferCandidate.from_offer(
                offer,
                as_of=AS_OF,
            ),
        ),
    )


def test_tool_requires_at_least_one_grounded_item() -> None:
    """The factory must not create a tool with an unrestricted item scope."""
    service = FakeGrocerySearchService(make_candidates())

    with pytest.raises(ValueError, match="at least one requested item"):
        create_find_flyer_deals_tool(
            service,
            requested_items=(),
        )


@pytest.mark.anyio
async def test_tool_preserves_grounded_item_when_searching_candidates() -> None:
    """The provider request should retain every validated shopping-item detail."""
    requested_item = ShoppingItem(
        name="broccoli",
        quantity=2,
        unit="crowns",
        notes="Use only the requested organic variety.",
        constraints=(
            ShoppingConstraint(
                value="organic",
                requirement=ConstraintRequirement.REQUIRED,
            ),
        ),
    )
    service = FakeGrocerySearchService(make_candidates())
    grocery_tool = create_find_flyer_deals_tool(
        service,
        requested_items=(requested_item,),
    )

    result = await grocery_tool.ainvoke(
        {
            "item_name": "broccoli",
            "postal_code": "l5b0g7",
            "store": None,
        }
    )

    assert grocery_tool.name == "find_flyer_deals"
    assert set(grocery_tool.args) == {"item_name", "postal_code", "store"}

    assert len(service.calls) == 1
    request, as_of = service.calls[0]
    assert request.item is requested_item
    assert request.item.quantity == 2
    assert request.item.unit == "crowns"
    assert request.item.notes == "Use only the requested organic variety."
    assert request.item.constraints == (
        ShoppingConstraint(
            value="organic",
            requirement=ConstraintRequirement.REQUIRED,
        ),
    )
    assert request.postal_code == "L5B 0G7"
    assert request.store is None
    assert isinstance(as_of, date)

    assert result["requested_item_name"] == "broccoli"
    assert result["provider_name"] == "fixture"
    assert "cheapest_offer" not in result
    assert "ranked_offers" not in result

    offer_payload = result["offers"]
    assert isinstance(offer_payload, list)
    assert len(offer_payload) == 1
    assert offer_payload[0]["store"] == "Nations Fresh Foods"
    assert offer_payload[0]["price"] == "1.29"
    assert offer_payload[0]["validity_status"] == "active"


@pytest.mark.anyio
async def test_tool_rejects_item_outside_grounded_request() -> None:
    """A hallucinated item must not reach the provider search boundary."""
    service = FakeGrocerySearchService(make_candidates())
    grocery_tool = create_find_flyer_deals_tool(
        service,
        requested_items=(ShoppingItem(name="broccoli"),),
    )

    result = await grocery_tool.ainvoke(
        {
            "item_name": "milk",
            "postal_code": "L5B 0G7",
            "store": None,
        }
    )

    assert service.calls == []
    assert result["status"] == "unsupported_item"
    assert result["requested_item_name"] == "milk"
    assert result["offers"] == []
    assert "No external search was performed" in str(result["reason"])


@pytest.mark.anyio
async def test_tool_rejects_ambiguous_grounded_item_name() -> None:
    """Same-name items with different constraints require explicit resolution."""
    service = FakeGrocerySearchService(make_candidates())
    grocery_tool = create_find_flyer_deals_tool(
        service,
        requested_items=(
            ShoppingItem(
                name="milk",
                constraints=(ShoppingConstraint(value="2%"),),
            ),
            ShoppingItem(
                name="milk",
                constraints=(ShoppingConstraint(value="lactose-free"),),
            ),
        ),
    )

    result = await grocery_tool.ainvoke(
        {
            "item_name": "milk",
            "postal_code": "L5B 0G7",
            "store": None,
        }
    )

    assert service.calls == []
    assert result["status"] == "unsupported_item"
    assert result["offers"] == []
    assert "multiple requested items" in str(result["reason"])
