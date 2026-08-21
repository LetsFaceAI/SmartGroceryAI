"""Offline tests for application-owned grocery search orchestration."""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from app.schemas.price_comparison import CheapestOfferStatus
from app.schemas.product_offer import MeasurementUnit, PriceBasis, ProductOffer
from app.schemas.search_provider import (
    GrocerySearchRequest,
    GrocerySearchResult,
    SearchProvenance,
)
from app.schemas.shopping import ShoppingItem
from app.services.grocery_search import GrocerySearchService

AS_OF = date(2026, 8, 20)


class FakeProvider:
    """Return controlled offers without network or provider SDK access."""

    def __init__(self, offers: tuple[ProductOffer, ...]) -> None:
        self._offers = offers
        self.requests: list[GrocerySearchRequest] = []

    @property
    def provider_name(self) -> str:
        return "fixture"

    async def search(
        self,
        request: GrocerySearchRequest,
    ) -> GrocerySearchResult:
        self.requests.append(request)
        return GrocerySearchResult(
            request=request,
            offers=self._offers,
            provenance=SearchProvenance(
                provider_name=self.provider_name,
                retrieved_at=datetime(2026, 8, 20, tzinfo=UTC),
            ),
        )


def make_coffee_offer(
    *,
    store: str,
    price: str,
    package_size: str,
    unit: MeasurementUnit,
) -> ProductOffer:
    """Build one active, comparable fixture offer."""
    return ProductOffer(
        product_name="Ground Coffee",
        store=store,
        price=Decimal(price),
        package_size=Decimal(package_size),
        unit=unit,
        price_basis=PriceBasis.TOTAL_PACKAGE,
        valid_from=date(2026, 8, 1),
        valid_until=date(2026, 8, 31),
        source=f"fixture:{store}",
    )


@pytest.mark.anyio
async def test_service_searches_and_selects_lowest_unit_price() -> None:
    """The service should rank normalized prices instead of shelf prices."""
    request = GrocerySearchRequest(
        item=ShoppingItem(name="ground coffee"),
        postal_code="M5V 3A8",
    )
    provider = FakeProvider(
        (
            make_coffee_offer(
                store="Corner Market",
                price="4.00",
                package_size="500",
                unit=MeasurementUnit.GRAM,
            ),
            make_coffee_offer(
                store="Warehouse Market",
                price="7.00",
                package_size="1",
                unit=MeasurementUnit.KILOGRAM,
            ),
        )
    )
    service = GrocerySearchService(provider)

    selection = await service.compare_offers(request, as_of=AS_OF)

    assert provider.requests == [request]
    assert selection.status is CheapestOfferStatus.SELECTED
    assert selection.cheapest_offer is not None
    assert selection.cheapest_offer.original_offer.store == "Warehouse Market"
    assert selection.cheapest_offer.unit_price == Decimal("7.000000000000")
