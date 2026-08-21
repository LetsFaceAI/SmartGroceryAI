"""Offline tests for the LangChain grocery coordinator."""

from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import patch

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import BaseTool, tool
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.runnables import Runnable

from app.agents.grocery_coordinator import (
    create_grocery_coordinator_agent,
    create_request_scoped_grocery_coordinator,
)
from app.core.config import Settings
from app.schemas.shopping import ShoppingItem, ShoppingRequest
from app.services.search_request_policy import ExternalActorCallBudget


class ToolCallingFakeChatModel(FakeMessagesListChatModel):
    """Allow deterministic fake responses to participate in tool binding."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


@pytest.mark.anyio
async def test_coordinator_executes_tool_call_and_returns_final_answer() -> None:
    """The agent should execute the requested tool before its final response."""
    calls: list[tuple[str, str, str | None]] = []

    @tool
    def find_flyer_deals(
        item_name: str,
        postal_code: str,
        store: str | None = None,
    ) -> str:
        """Retrieve validated flyer candidates for one item."""
        calls.append((item_name, postal_code, store))
        return (
            '{"requested_item_name":"ground coffee","offers":['
            '{"store":"Warehouse Market","price":"7.00","currency":"CAD",'
            '"price_basis":"unknown","validity_status":"active"}]}'
        )

    model = ToolCallingFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_flyer_deals",
                        "args": {
                            "item_name": "ground coffee",
                            "postal_code": "M5V 3A8",
                        },
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "Warehouse Market is the best available advertised offer "
                    "at CAD 7.00. The missing price basis means this is not a "
                    "verified unit-price comparison."
                )
            ),
        ]
    )
    agent = create_grocery_coordinator_agent(
        model,
        [find_flyer_deals],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Find coffee deals near M5V 3A8.")]}
    )

    assert calls == [("ground coffee", "M5V 3A8", None)]

    messages = result["messages"]
    assert [type(message) for message in messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert messages[2].content == (
        '{"requested_item_name":"ground coffee","offers":['
        '{"store":"Warehouse Market","price":"7.00","currency":"CAD",'
        '"price_basis":"unknown","validity_status":"active"}]}'
    )
    assert messages[-1].content == (
        "Warehouse Market is the best available advertised offer "
        "at CAD 7.00. The missing price basis means this is not a "
        "verified unit-price comparison."
    )


def test_request_scoped_coordinators_receive_independent_budgets() -> None:
    """Each shopping request should own a fresh paid-call allowance."""
    settings = Settings(
        _env_file=None,
        apify_mcp_tool_timeout_seconds=15,
        search_max_external_actor_calls_per_request=1,
        search_max_concurrency=1,
    )  # type: ignore[call-arg]
    request = ShoppingRequest(
        items=[ShoppingItem(name="milk")],
    )
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="unused")],
    )

    with patch("app.agents.grocery_coordinator.ApifyFlippProvider") as provider_factory:
        first_agent = create_request_scoped_grocery_coordinator(
            request,
            model=model,
            settings=settings,
        )
        second_agent = create_request_scoped_grocery_coordinator(
            request,
            model=model,
            settings=settings,
        )

    assert first_agent is not second_agent
    assert provider_factory.call_count == 2

    first_provider_call = provider_factory.call_args_list[0]
    second_provider_call = provider_factory.call_args_list[1]

    first_budget = first_provider_call.kwargs["call_budget"]
    second_budget = second_provider_call.kwargs["call_budget"]

    assert isinstance(first_budget, ExternalActorCallBudget)
    assert isinstance(second_budget, ExternalActorCallBudget)
    assert first_budget is not second_budget
    assert first_budget.remaining_calls == 1
    assert second_budget.remaining_calls == 1

    assert first_provider_call.kwargs["timeout_seconds"] == 15
    assert second_provider_call.kwargs["timeout_seconds"] == 15

    with first_budget.actor_call():
        pass

    assert first_budget.remaining_calls == 0
    assert second_budget.remaining_calls == 1
