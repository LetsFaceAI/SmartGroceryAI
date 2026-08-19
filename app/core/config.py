"""Central, environment-based configuration for SmartGroceryAI.

Pydantic Settings reads operating-system environment variables and an optional
local ``.env`` file, applies defaults, and validates the final values. Application
code should use :func:`get_settings` instead of reading environment variables
directly so configuration remains consistent and easy to extend.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Define every supported application setting and its validation rules.

    Python attributes use snake_case, while ``validation_alias`` documents the
    corresponding uppercase environment-variable name.
    """

    # Environment variables take precedence over values in .env. Unknown .env
    # entries are ignored so one shared file can later contain service settings.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        populate_by_name=True,
    )

    # A non-empty display name is useful in logs and future API metadata.
    app_name: str = Field(
        default="SmartGroceryAI",
        min_length=1,
        validation_alias="APP_NAME",
    )
    # Restrict environments to known deployment stages to catch typing mistakes.
    app_env: Literal["development", "test", "staging", "production"] = Field(
        default="development",
        validation_alias="APP_ENV",
    )
    # Accept only levels supported by Python's standard logging package.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
    )
    # SecretStr masks the credential in logs and object representations. None keeps
    # non-LLM commands, such as linting and tests, usable without an API account.
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    # The model is configurable so environments can trade off quality, latency,
    # and cost without changing application code.
    openai_model: str = Field(
        default="gpt-5-nano",
        min_length=1,
        validation_alias="OPENAI_MODEL",
    )
    # The URL can include Apify's tool allowlist query string, letting deployment
    # configuration expose only the selected Flipp Actor to the MCP client.
    apify_mcp_server_url: AnyHttpUrl | None = Field(
        default=None,
        validation_alias="APIFY_MCP_SERVER_URL",
    )
    # The token is separate from the URL so it is sent in an Authorization header
    # and remains masked in settings representations and logs.
    apify_api_token: SecretStr | None = Field(
        default=None,
        validation_alias="APIFY_API_TOKEN",
    )


@lru_cache
def get_settings() -> Settings:
    """Load and cache the validated application settings.

    Reading configuration once keeps every caller consistent and avoids repeatedly
    parsing the environment. Tests that change environment variables can call
    ``get_settings.cache_clear()`` before requesting a fresh instance.
    """
    return Settings()
