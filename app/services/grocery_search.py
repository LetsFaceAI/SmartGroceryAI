"""Application-owned grocery search and comparison orchestration."""

from datetime import date

from app.providers.base import GrocerySearchProvider
from app.schemas.price_comparison import CheapestOfferSelection
from app.schemas.search_provider import GrocerySearchRequest
from app.services.price_comparison import select_cheapest_offer
from app.services.product_matcher import build_product_match_request, match_product
from app.services.product_normalization import normalize_product_offer


class GrocerySearchService:
    """Search for offers and compare them using deterministic domain logic."""

    def __init__(self, provider: GrocerySearchProvider) -> None:
        self._provider = provider

    async def compare_offers(
        self,
        request: GrocerySearchRequest,
        *,
        as_of: date,
    ) -> CheapestOfferSelection:
        """Return the cheapest trusted, active, comparable offer."""
        search_result = await self._provider.search(request)
        match_request = build_product_match_request(request.item)

        match_results = tuple(
            match_product(
                match_request,
                normalize_product_offer(offer),
            )
            for offer in search_result.offers
        )

        return select_cheapest_offer(
            request.item.name,
            match_results,
            as_of=as_of,
        )