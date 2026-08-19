"""Prompt and message construction for the grocery-assistant conversation flow.

This module owns what the assistant should do, while ``app.core.llm`` owns how a
model is configured and invoked. Keeping those responsibilities separate makes
prompt changes testable without credentials or network access.
"""

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage

GROCERY_ASSISTANT_SYSTEM_PROMPT = """\
You are SmartGroceryAI, an assistant focused only on grocery shopping.

Help users interpret grocery lists, compare grocery deals when deal data is
provided, identify lower-cost choices, suggest practical grocery alternatives,
and organize purchases by store.

Do not invent prices, sales, store availability, or current or upcoming flyer
deals. Clearly say when current deal data is required. If a request is unrelated
to grocery shopping, briefly explain that you can only help with grocery-related
questions. Keep answers clear, practical, and concise.
"""


def build_grocery_messages(user_input: str) -> list[BaseMessage]:
    """Build one grocery-assistant turn from system and user messages.

    Args:
        user_input: The user's natural-language grocery request.

    Returns:
        A system message defining assistant behavior followed by the human message.

    Raises:
        ValueError: If the user input is empty or contains only whitespace.
    """
    normalized_input = user_input.strip()
    if not normalized_input:
        raise ValueError("A grocery-assistant message cannot be empty.")

    return [
        SystemMessage(content=GROCERY_ASSISTANT_SYSTEM_PROMPT),
        HumanMessage(content=normalized_input),
    ]
