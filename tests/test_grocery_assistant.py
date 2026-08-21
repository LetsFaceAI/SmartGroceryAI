"""Unit tests for grocery-assistant prompt and message construction."""

import pytest
from langchain.messages import HumanMessage, SystemMessage

from app.prompts.grocery_assistant import (
    GROCERY_ASSISTANT_SYSTEM_PROMPT,
    GROCERY_COORDINATOR_SYSTEM_PROMPT,
    SHOPPING_REQUEST_EXTRACTION_SYSTEM_PROMPT,
    build_grocery_messages,
    build_shopping_request_messages,
)


def test_build_grocery_messages_creates_ordered_prompt_turn() -> None:
    """The system instruction should precede normalized user input."""
    messages = build_grocery_messages("  Find a cheaper alternative to milk.  ")

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == GROCERY_ASSISTANT_SYSTEM_PROMPT
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "Find a cheaper alternative to milk."


def test_build_grocery_messages_rejects_empty_input() -> None:
    """Empty requests should fail locally instead of reaching a paid model call."""
    with pytest.raises(ValueError, match="cannot be empty"):
        build_grocery_messages("   ")


def test_build_shopping_request_messages_adds_extraction_rules() -> None:
    """Structured extraction should add precise rules without changing user text."""
    messages = build_shopping_request_messages("  I prefer organic 2% milk.  ")

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == SHOPPING_REQUEST_EXTRACTION_SYSTEM_PROMPT
    assert "Include only grocery items" in str(messages[0].content)
    assert "request-wide preferences" in str(messages[0].content)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "I prefer organic 2% milk."


def test_coordinator_prompt_delegates_candidate_selection_to_model() -> None:
    """The coordinator should select candidates while qualifying uncertainty."""
    assert "best available advertised offer" in GROCERY_COORDINATOR_SYSTEM_PROMPT
    assert "they are not ranked" in GROCERY_COORDINATOR_SYSTEM_PROMPT
    assert "missing or incompatible price bases" in (GROCERY_COORDINATOR_SYSTEM_PROMPT)
    assert "Do not call a store the closest or nearest" in (
        GROCERY_COORDINATOR_SYSTEM_PROMPT
    )
    assert "ranking as authoritative" not in GROCERY_COORDINATOR_SYSTEM_PROMPT
    assert "Do not recalculate prices" not in GROCERY_COORDINATOR_SYSTEM_PROMPT
