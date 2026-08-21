"""LangChain tools for application-owned grocery search capabilities."""

from collections.abc import Sequence
from datetime import date

from langchain.tools import BaseTool, tool

from app.schemas.search_provider import GrocerySearchRequest
from app.schemas.shopping import ShoppingItem
from app.services.grocery_search import GrocerySearchService
from app.services.product_normalization import normalize_product_name


def _unsupported_item_result(
    item_name: str,
    *,
    reason: str,
) -> dict[str, object]:
    """Return a safe tool result without performing an external search."""
    return {
        "status": "unsupported_item",
        "requested_item_name": item_name.strip(),
        "reason": reason,
        "offers": [],
    }


def create_find_flyer_deals_tool(
    service: GrocerySearchService,
    *,
    requested_items: Sequence[ShoppingItem],
) -> BaseTool:
    """Create a flyer tool grounded to one request's validated shopping items."""
    item_lookup: dict[str, list[ShoppingItem]] = {}
    for requested_item in requested_items:
        normalized_name = normalize_product_name(requested_item.name)
        item_lookup.setdefault(normalized_name, []).append(requested_item)

    if not item_lookup:
        raise ValueError("The flyer tool requires at least one requested item.")

    @tool
    async def find_flyer_deals(
        item_name: str,
        postal_code: str,
        store: str | None = None,
    ) -> dict[str, object]:
        """Retrieve flyer candidates for an item in the grounded user request."""
        matching_items = item_lookup.get(normalize_product_name(item_name), [])
        if not matching_items:
            return _unsupported_item_result(
                item_name,
                reason=(
                    "The item is not part of the grounded shopping request. "
                    "No external search was performed."
                ),
            )

        if len(matching_items) > 1:
            return _unsupported_item_result(
                item_name,
                reason=(
                    "The item name matches multiple requested items with different "
                    "details. No external search was performed."
                ),
            )

        request = GrocerySearchRequest(
            item=matching_items[0],
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
