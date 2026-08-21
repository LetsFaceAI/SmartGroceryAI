"""Single-agent coordinator for grocery conversations and tool use."""

from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from app.core.config import Settings, get_settings
from app.core.llm import create_chat_model
from app.prompts.grocery_assistant import GROCERY_COORDINATOR_SYSTEM_PROMPT
from app.providers.apify_flipp import ApifyFlippProvider
from app.schemas.shopping import ShoppingRequest
from app.services.grocery_search import GrocerySearchService
from app.services.search_request_policy import (
    ExternalActorCallBudget,
    build_search_request_plan,
)
from app.tools.grocery_search import create_find_flyer_deals_tool


def create_grocery_coordinator_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
) -> CompiledStateGraph:
    """Create the coordinator with its model, tools, and behavioral prompt."""
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=GROCERY_COORDINATOR_SYSTEM_PROMPT,
    )


def create_request_scoped_grocery_coordinator(
    request: ShoppingRequest,
    *,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Compose one coordinator with a fresh provider budget."""
    resolved_settings = settings or get_settings()

    search_plan = build_search_request_plan(
        request,
        settings=resolved_settings,
    )
    call_budget = ExternalActorCallBudget(search_plan)

    provider = ApifyFlippProvider(
        call_budget=call_budget,
        timeout_seconds=resolved_settings.apify_mcp_tool_timeout_seconds,
    )
    service = GrocerySearchService(provider)
    flyer_tool = create_find_flyer_deals_tool(
        service,
        requested_items=search_plan.items,
    )

    resolved_model = (
        model if model is not None else create_chat_model(resolved_settings)
    )
    return create_grocery_coordinator_agent(
        resolved_model,
        [flyer_tool],
    )
