"""Offline tests for the LangChain grocery coordinator."""

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import BaseTool, tool
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.runnables import Runnable

from app.agents.grocery_coordinator import create_grocery_coordinator_agent


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
        """Search local grocery flyers and compare current deals for one item."""
        calls.append((item_name, postal_code, store))
        return "Warehouse Market: CAD 7.00 per kg."

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
                    "Warehouse Market has the cheapest ground coffee "
                    "at CAD 7.00 per kg."
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
    assert messages[2].content == "Warehouse Market: CAD 7.00 per kg."
    assert messages[-1].content == (
        "Warehouse Market has the cheapest ground coffee at CAD 7.00 per kg."
    )
