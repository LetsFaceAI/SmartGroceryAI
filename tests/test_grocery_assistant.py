"""Unit tests for grocery-assistant prompt and message construction."""

import pytest
from langchain.messages import HumanMessage, SystemMessage

from app.prompts.grocery_assistant import (
    GROCERY_ASSISTANT_SYSTEM_PROMPT,
    build_grocery_messages,
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
