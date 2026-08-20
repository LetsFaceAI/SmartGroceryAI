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
        """Search local grocery flyers and compare current deals for one item."""
        request = GrocerySearchRequest(
            item=ShoppingItem(name=item_name),
            postal_code=postal_code,
            store=store,
        )
        selection = await service.compare_offers(
            request,
            as_of=date.today(),
        )
        return selection.model_dump(mode="json")

    return find_flyer_deals