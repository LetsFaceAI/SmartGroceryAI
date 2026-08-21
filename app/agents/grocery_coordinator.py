"""Single-agent coordinator for grocery conversations and tool use."""

from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from app.prompts.grocery_assistant import GROCERY_COORDINATOR_SYSTEM_PROMPT


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
