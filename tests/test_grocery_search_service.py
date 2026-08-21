"""Offline tests for grocery offer retrieval and candidate curation."""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from app.schemas.offer_candidates import FlyerOfferValidity
from app.schemas.product_offer import (
    MeasurementUnit,
    PriceBasis,
    ProductOffer,
)
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


def make_broccoli_offer(
    *,
    store: str,
    price: str,
    source: str,
    valid_from: date = date(2026, 8, 14),
    valid_until: date = date(2026, 8, 20),
    price_basis: PriceBasis = PriceBasis.UNKNOWN,
    price_basis_unit: MeasurementUnit | None = None,
) -> ProductOffer:
    """Build one realistic broccoli fixture returned by the provider."""
    return ProductOffer(
        product_name="Crown Broccoli",
        store=store,
        price=Decimal(price),
        price_basis=price_basis,
        price_basis_unit=price_basis_unit,
        valid_from=valid_from,
        valid_until=valid_until,
        source=source,
    )


@pytest.mark.anyio
async def test_service_builds_unranked_deduplicated_candidates() -> None:
    """The service should curate candidates without calculating a winner."""
    request = GrocerySearchRequest(
        item=ShoppingItem(name="broccoli"),
        postal_code="L5B 0G7",
    )
    provider = FakeProvider(
        (
            make_broccoli_offer(
                store="Nations Fresh Foods",
                price="1.29",
                source="fixture:nations:first",
            ),
            make_broccoli_offer(
                store="Nations Fresh Foods",
                price="1.29",
                source="fixture:nations:duplicate",
            ),
            make_broccoli_offer(
                store="Bestco Foodmart",
                price="1.49",
                price_basis=PriceBasis.PER_WEIGHT,
                price_basis_unit=MeasurementUnit.POUND,
                source="fixture:bestco",
            ),
            make_broccoli_offer(
                store="Btrust Supermarket",
                price="1.28",
                valid_from=date(2026, 8, 21),
                valid_until=date(2026, 8, 27),
                source="fixture:btrust",
            ),
        )
    )
    service = GrocerySearchService(provider)

    candidates = await service.search_offer_candidates(
        request,
        as_of=AS_OF,
    )

    assert provider.requests == [request]
    assert candidates.requested_item_name == "broccoli"
    assert candidates.postal_code == "L5B 0G7"
    assert candidates.provider_name == "fixture"
    assert len(candidates.offers) == 3

    offers_by_store = {offer.store: offer for offer in candidates.offers}
    assert (
        offers_by_store["Nations Fresh Foods"].validity_status
        is FlyerOfferValidity.ACTIVE
    )
    assert offers_by_store["Nations Fresh Foods"].source == ("fixture:nations:first")
    assert offers_by_store["Bestco Foodmart"].price_basis is PriceBasis.PER_WEIGHT
    assert offers_by_store["Bestco Foodmart"].price_basis_unit is MeasurementUnit.POUND
    assert (
        offers_by_store["Btrust Supermarket"].validity_status
        is FlyerOfferValidity.UPCOMING
    )

    payload = candidates.model_dump(mode="json")
    assert "cheapest_offer" not in payload
    assert "ranked_offers" not in payload


@pytest.mark.anyio
async def test_service_can_return_validated_provider_result_directly() -> None:
    """Low-level retrieval should retain the validated provider boundary."""
    request = GrocerySearchRequest(
        item=ShoppingItem(name="broccoli"),
        postal_code="L5B 0G7",
    )
    offers = (
        make_broccoli_offer(
            store="Nations Fresh Foods",
            price="1.29",
            source="fixture:nations",
        ),
    )
    provider = FakeProvider(offers)
    service = GrocerySearchService(provider)

    result = await service.search_offers(request)

    assert provider.requests == [request]
    assert result.request is request
    assert result.offers == offers
    assert result.provenance.provider_name == "fixture"
