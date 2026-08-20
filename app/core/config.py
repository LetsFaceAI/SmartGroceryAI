"""Central, environment-based configuration for SmartGroceryAI.

Pydantic Settings reads operating-system environment variables and an optional
local ``.env`` file, applies defaults, and validates the final values. Application
code should use :func:`get_settings` instead of reading environment variables
directly so configuration remains consistent and easy to extend.
"""

from functools import lru_cache
from typing import Literal
from urllib.parse import parse_qsl

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Only parameters documented by Apify's hosted MCP server may cross this
# configuration boundary. An allowlist is safer than trying to enumerate every
# possible credential-like spelling that must be rejected.
APIFY_MCP_ALLOWED_QUERY_PARAMETERS = frozenset(
    {"actors", "telemetry-enabled", "tools", "ui"}
)


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
    # Bound one discovery-plus-execution attempt. The application never retries a
    # paid Actor call automatically, even when this client-side timeout expires.
    apify_mcp_tool_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=600,
        validation_alias="APIFY_MCP_TOOL_TIMEOUT_SECONDS",
    )
    # These limits are application policy, not Actor capabilities. Keeping them in
    # validated configuration prevents future agent prompts from increasing spend.
    search_max_items_per_request: int = Field(
        default=20,
        ge=1,
        le=100,
        validation_alias="SEARCH_MAX_ITEMS_PER_REQUEST",
    )
    search_max_external_actor_calls_per_request: int = Field(
        default=3,
        ge=1,
        le=10,
        validation_alias="SEARCH_MAX_EXTERNAL_ACTOR_CALLS_PER_REQUEST",
    )
    search_max_concurrency: int = Field(
        default=1,
        ge=1,
        le=5,
        validation_alias="SEARCH_MAX_CONCURRENCY",
    )

    @field_validator("apify_mcp_server_url")
    @classmethod
    def validate_apify_mcp_server_url(
        cls,
        value: AnyHttpUrl | None,
    ) -> AnyHttpUrl | None:
        """Keep the Apify credential restricted to its official HTTPS endpoint.

        The discovery client attaches ``APIFY_API_TOKEN`` to this URL. Restricting
        the scheme and host prevents a configuration mistake from forwarding that
        credential to an unrelated server.
        """
        if value is not None and (
            value.scheme != "https" or value.host != "mcp.apify.com"
        ):
            raise ValueError("APIFY_MCP_SERVER_URL must use https://mcp.apify.com.")
        if value is not None and (
            value.username is not None or value.password is not None
        ):
            raise ValueError(
                "APIFY_MCP_SERVER_URL must not contain credentials; use the "
                "dedicated APIFY_API_TOKEN setting instead."
            )

        if value is not None:
            query_parameters = {
                key
                for key, _ in parse_qsl(
                    value.query or "",
                    keep_blank_values=True,
                )
            }
            if not query_parameters.issubset(APIFY_MCP_ALLOWED_QUERY_PARAMETERS):
                # Do not include rejected names or values in this message. A
                # misspelled credential parameter may itself reveal a secret.
                allowed = ", ".join(sorted(APIFY_MCP_ALLOWED_QUERY_PARAMETERS))
                raise ValueError(
                    "APIFY_MCP_SERVER_URL contains an unsupported query parameter. "
                    f"Allowed parameters: {allowed}."
                )
        return value


@lru_cache
def get_settings() -> Settings:
    """Load and cache the validated application settings.

    Reading configuration once keeps every caller consistent and avoids repeatedly
    parsing the environment. Tests that change environment variables can call
    ``get_settings.cache_clear()`` before requesting a fresh instance.
    """
    return Settings()
