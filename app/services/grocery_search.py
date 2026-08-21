"""Application-owned grocery offer retrieval and candidate curation."""

from collections.abc import Sequence
from datetime import date

from app.providers.base import GrocerySearchProvider
from app.schemas.offer_candidates import (
    FlyerOfferCandidate,
    FlyerOfferCandidateSet,
)
from app.schemas.product_offer import ProductOffer
from app.schemas.search_provider import (
    GrocerySearchRequest,
    GrocerySearchResult,
)


def _normalized_optional_text(value: str | None) -> str | None:
    """Normalize text used only for duplicate detection."""
    if value is None:
        return None
    return " ".join(value.casefold().split())


def _offer_identity(offer: ProductOffer) -> tuple[object, ...]:
    """Identify equivalent offers while deliberately ignoring their source IDs."""
    return (
        _normalized_optional_text(offer.product_name),
        _normalized_optional_text(offer.brand),
        _normalized_optional_text(offer.store),
        offer.price,
        offer.regular_price,
        offer.currency,
        offer.package_size,
        offer.package_quantity,
        offer.unit,
        offer.price_basis,
        offer.price_basis_unit,
        offer.promotion_status,
        offer.valid_from,
        offer.valid_until,
    )


def _deduplicate_offers(
    offers: Sequence[ProductOffer],
) -> tuple[ProductOffer, ...]:
    """Preserve the first copy of each equivalent validated offer."""
    unique_offers: list[ProductOffer] = []
    seen: set[tuple[object, ...]] = set()

    for offer in offers:
        identity = _offer_identity(offer)
        if identity in seen:
            continue

        seen.add(identity)
        unique_offers.append(offer)

    return tuple(unique_offers)


class GrocerySearchService:
    """Retrieve validated offers without calculating or selecting a winner."""

    def __init__(self, provider: GrocerySearchProvider) -> None:
        self._provider = provider

    async def search_offers(
        self,
        request: GrocerySearchRequest,
    ) -> GrocerySearchResult:
        """Return validated provider offers without selecting a winner."""
        return await self._provider.search(request)

    async def search_offer_candidates(
        self,
        request: GrocerySearchRequest,
        *,
        as_of: date,
    ) -> FlyerOfferCandidateSet:
        """Return compact, deduplicated candidates for coordinator reasoning."""
        result = await self.search_offers(request)
        unique_offers = _deduplicate_offers(result.offers)

        return FlyerOfferCandidateSet(
            requested_item_name=request.item.name,
            postal_code=request.postal_code,
            store=request.store,
            as_of=as_of,
            provider_name=result.provenance.provider_name,
            retrieved_at=result.provenance.retrieved_at,
            offers=tuple(
                FlyerOfferCandidate.from_offer(
                    offer,
                    as_of=as_of,
                )
                for offer in unique_offers
            ),
        )
