"""Reusable service for one-turn grocery-assistant conversations.

The service coordinates the grocery prompt and LangChain chat model, then converts
the provider-neutral ``AIMessage`` into plain text for application callers. Model
configuration remains in ``app.core.llm`` and prompt behavior remains in
``app.prompts``.
"""

from langchain.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.llm import create_chat_model
from app.prompts.grocery_assistant import build_grocery_messages


class GroceryAssistantServiceError(RuntimeError):
    """Provide a stable application error for failed assistant operations."""


class EmptyModelResponseError(GroceryAssistantServiceError):
    """Report a successful invocation that contains no usable assistant text."""


def get_grocery_assistant_response(
    user_message: str,
    model: BaseChatModel | None = None,
) -> str:
    """Return the assistant's text response for one grocery-related user message.

    Args:
        user_message: Natural-language grocery request from the user.
        model: Optional chat model. Supplying one supports isolated tests and future
            composition; otherwise the centralized model factory is used.

    Returns:
        Non-empty assistant text with surrounding whitespace removed.

    Raises:
        ValueError: If ``user_message`` is empty or whitespace-only.
        LLMConfigurationError: If the default model lacks required configuration.
        GroceryAssistantServiceError: If invocation fails or returns an unexpected
            message type.
        EmptyModelResponseError: If the model returns no usable text.
    """
    # Prompt validation happens before model creation or invocation, preventing an
    # invalid request from reaching a paid external service.
    messages = build_grocery_messages(user_message)
    chat_model = model if model is not None else create_chat_model()

    try:
        response = chat_model.invoke(messages)
    except Exception as exc:
        # Preserve the provider exception as the cause for logs and debugging while
        # exposing a stable service-level error to future API or UI callers.
        raise GroceryAssistantServiceError(
            "The grocery assistant model invocation failed."
        ) from exc

    if not isinstance(response, AIMessage):
        raise GroceryAssistantServiceError(
            "The grocery assistant returned an unexpected message type."
        )

    assistant_text = response.text.strip()
    if not assistant_text:
        raise EmptyModelResponseError(
            "The grocery assistant returned an empty response."
        )

    return assistant_text
