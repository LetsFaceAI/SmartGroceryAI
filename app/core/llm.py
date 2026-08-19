"""Central LangChain chat-model creation and minimal invocation helpers.

Only this module knows which model provider is currently used. Feature code can
depend on LangChain's provider-neutral ``BaseChatModel`` interface, allowing model
parameters or providers to change later without spreading construction logic across
the application.
"""

from langchain.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings
from app.prompts.grocery_assistant import build_grocery_messages


class LLMConfigurationError(RuntimeError):
    """Report missing or unusable configuration before an API request is attempted."""


def create_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Create the configured LangChain chat model.

    Args:
        settings: Optional settings instance, primarily useful for tests. When it is
            omitted, the process-wide application settings are used.

    Returns:
        A provider-neutral LangChain chat-model interface backed by OpenAI.

    Raises:
        LLMConfigurationError: If ``OPENAI_API_KEY`` is missing or blank.
    """
    resolved_settings = settings or get_settings()
    api_key = resolved_settings.openai_api_key

    if api_key is None or not api_key.get_secret_value().strip():
        raise LLMConfigurationError(
            "OPENAI_API_KEY is not configured. Copy .env.example to .env and add "
            "your OpenAI API key before making a model request."
        )

    return ChatOpenAI(
        model=resolved_settings.openai_model,
        api_key=api_key,
    )


def send_message(
    message: str,
    model: BaseChatModel | None = None,
) -> AIMessage:
    """Send one grocery-assistant prompt turn and return its AI response.

    Args:
        message: Natural-language input to send to the model.
        model: Optional model instance. The configured model is created when omitted;
            injection keeps tests offline and supports future model composition.

    Returns:
        The model's response as a LangChain ``AIMessage`` with content and metadata.

    Raises:
        LLMConfigurationError: If a model must be created without an API key.
        TypeError: If a provider returns an unexpected message type.
    """
    chat_model = model or create_chat_model()
    messages = build_grocery_messages(message)
    response = chat_model.invoke(messages)

    # Chat models should produce AIMessage objects. Keep this boundary explicit so
    # later code can safely access response metadata, content blocks, and tool calls.
    if not isinstance(response, AIMessage):
        raise TypeError(
            f"Expected an AIMessage response, received {type(response).__name__}."
        )

    return response
