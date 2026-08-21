"""LangChain tools for application-owned grocery search capabilities."""

from datetime import date

from langchain.tools import BaseTool, tool

from app.schemas.search_provider import GrocerySearchRequest
from app.schemas.shopping import ShoppingItem
from app.services.grocery_search import GrocerySearchService


def create_find_flyer_deals_tool(
    service: GrocerySearchService,
) -> BaseTool:
    """Create a flyer-search tool backed by the application service."""

    @tool
    async def find_flyer_deals(
        item_name: str,
        postal_code: str,
        store: str | None = None,
    ) -> dict[str, object]:
        """Retrieve validated local flyer candidates for model-guided selection."""
        request = GrocerySearchRequest(
            item=ShoppingItem(name=item_name),
            postal_code=postal_code,
            store=store,
        )
        candidates = await service.search_offer_candidates(
            request,
            as_of=date.today(),
        )
        return candidates.model_dump(
            mode="json",
            exclude_none=True,
        )

    return find_flyer_deals
