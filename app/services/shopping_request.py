"""Structured extraction service for natural-language grocery requests.

This service composes three existing boundaries: the grocery prompt defines model
behavior, the centralized factory supplies a provider-backed chat model, and the
``ShoppingRequest`` schema defines valid output. No tool or agent behavior belongs
in this extraction step.
"""

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import ValidationError

from app.core.llm import create_chat_model
from app.prompts.grocery_assistant import build_grocery_messages
from app.schemas.shopping import ShoppingRequest


class ShoppingRequestExtractionError(RuntimeError):
    """Report model setup or invocation failures during request extraction."""


class ShoppingRequestValidationError(ShoppingRequestExtractionError):
    """Report model output that cannot satisfy the shopping request schema."""


def extract_shopping_request(
    user_input: str,
    model: BaseChatModel | None = None,
) -> ShoppingRequest:
    """Extract a validated shopping request from natural-language input.

    Args:
        user_input: A grocery request such as "I need two bags of milk and eggs."
        model: Optional base chat model. Injection keeps unit tests offline; when it
            is omitted, the centralized application model factory is used.

    Returns:
        A fully validated ``ShoppingRequest`` instance.

    Raises:
        ValueError: If ``user_input`` is empty or whitespace-only.
        LLMConfigurationError: If the default model lacks required configuration.
        ShoppingRequestValidationError: If model output cannot satisfy the schema.
        ShoppingRequestExtractionError: If structured-model setup or invocation
            fails for another reason.
    """
    # Validate and construct the prompt before creating a model so empty input never
    # reaches an external provider or incurs usage.
    messages = build_grocery_messages(user_input)
    chat_model = model if model is not None else create_chat_model()

    try:
        # LangChain binds the Pydantic schema to the model and parses the response
        # through that schema instead of asking callers to process free-form text.
        structured_model = chat_model.with_structured_output(ShoppingRequest)
        result = structured_model.invoke(messages)

        # Provider integrations normally return ShoppingRequest directly. The
        # fallback validation keeps this boundary safe for integrations returning a
        # dictionary-like payload while still enforcing the exact same schema.
        if isinstance(result, ShoppingRequest):
            return result
        return ShoppingRequest.model_validate(result)
    except (ValidationError, OutputParserException) as exc:
        raise ShoppingRequestValidationError(
            "The model response did not match the ShoppingRequest schema."
        ) from exc
    except Exception as exc:
        # Retain the underlying provider error for logs while exposing a stable error
        # type that a future API or UI can handle without provider-specific imports.
        raise ShoppingRequestExtractionError(
            "The shopping request model invocation failed."
        ) from exc
