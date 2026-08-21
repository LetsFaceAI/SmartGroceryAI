"""Prompt and message construction for the grocery-assistant conversation flow.

This module owns what the assistant should do, while ``app.core.llm`` owns model
configuration and the service layer owns invocation. Keeping those responsibilities
separate makes prompt changes testable without credentials or network access.
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

# Structured extraction needs narrower instructions than a conversational response.
# These rules reduce guesswork while Pydantic remains the final validation boundary.
SHOPPING_REQUEST_EXTRACTION_SYSTEM_PROMPT = (
    GROCERY_ASSISTANT_SYSTEM_PROMPT
    + """

For this task, extract the user's message into the provided shopping request schema.
Follow these extraction rules:
- Include only grocery items or grocery categories explicitly mentioned by the user.
- Use the schema default quantity when no quantity is stated.
- Include a unit only when the user states one; otherwise leave it unset.
- Put item-specific qualifiers such as "2%", "lactose-free", "gluten-free", or
  "organic" in that item's constraints. Mark a constraint optional only when the
  user clearly uses language such as "prefer", "if possible", or "when available";
  otherwise mark it required. Keep other item context in notes.
- Put preferences that apply to the whole request, such as "organic when possible"
  or "lowest price", in the request-wide preferences list.
- Preserve vague categories such as "fruit" instead of inventing a specific item.
- Do not invent brands, quantities, units, preferences, or missing grocery items.
"""
)

GROCERY_COORDINATOR_SYSTEM_PROMPT = (
    GROCERY_ASSISTANT_SYSTEM_PROMPT
    + """

You can use application-owned tools to retrieve current grocery flyer data.

When a user asks for current, local, or store-specific flyer deals:
- Call find_flyer_deals instead of answering from memory.
- Ask for a postal code if the user has not provided one.
- Make one tool call per requested grocery item.
- Use only stores, prices, validity, and sources returned by the tool.
- Treat the tool's matching, unit-price calculations, and ranking as authoritative.
- Do not recalculate prices or override the deterministic ranking.
- Clearly explain when the tool returns no comparable offer.
"""
)


def _normalize_user_input(user_input: str) -> str:
    """Trim and validate user text before any prompt reaches a model."""
    normalized_input = user_input.strip()
    if not normalized_input:
        raise ValueError("A grocery-assistant message cannot be empty.")
    return normalized_input


def build_grocery_messages(user_input: str) -> list[BaseMessage]:
    """Build one grocery-assistant turn from system and user messages.

    Args:
        user_input: The user's natural-language grocery request.

    Returns:
        A system message defining assistant behavior followed by the human message.

    Raises:
        ValueError: If the user input is empty or contains only whitespace.
    """
    normalized_input = _normalize_user_input(user_input)

    return [
        SystemMessage(content=GROCERY_ASSISTANT_SYSTEM_PROMPT),
        HumanMessage(content=normalized_input),
    ]


def build_shopping_request_messages(user_input: str) -> list[BaseMessage]:
    """Build messages specifically for schema-backed shopping request extraction.

    Args:
        user_input: Natural-language grocery input to normalize and extract.

    Returns:
        Extraction instructions followed by the normalized human input.

    Raises:
        ValueError: If the user input is empty or contains only whitespace.
    """
    normalized_input = _normalize_user_input(user_input)

    return [
        SystemMessage(content=SHOPPING_REQUEST_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=normalized_input),
    ]
