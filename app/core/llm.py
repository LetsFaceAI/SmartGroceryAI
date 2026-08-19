"""Central LangChain chat-model configuration and creation.

Only this module knows which model provider is currently used. Feature code can
depend on LangChain's provider-neutral ``BaseChatModel`` interface, allowing model
parameters or providers to change later without spreading construction logic across
the application.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings


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
