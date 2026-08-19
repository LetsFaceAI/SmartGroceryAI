"""Unit tests for offline Apify MCP client configuration."""

from unittest.mock import Mock

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import ValidationError

from app.core.config import Settings
from app.core.mcp import (
    APIFY_FLIPP_SERVER_NAME,
    MCPConfigurationError,
    create_apify_mcp_client,
)


def make_settings_without_env(settings_data: dict[str, object]) -> Settings:
    """Build deterministic settings without reading a developer's real .env file."""
    # Pydantic Settings supports _env_file at runtime, but its generated
    # constructor signature does not expose this test-only option to mypy.
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        **settings_data,  # type: ignore[arg-type]
    )


def test_apify_mcp_server_url_requires_a_valid_http_url() -> None:
    """Malformed endpoint configuration should fail during settings validation."""
    with pytest.raises(ValidationError, match="apify_mcp_server_url"):
        make_settings_without_env({"apify_mcp_server_url": "not-a-url"})


@pytest.mark.parametrize(
    "server_url",
    [
        "http://mcp.apify.com",
        "https://example.com/mcp",
    ],
)
def test_apify_mcp_server_url_restricts_where_token_can_be_sent(
    server_url: str,
) -> None:
    """Only Apify's official HTTPS MCP host may receive the API token."""
    with pytest.raises(ValidationError, match="https://mcp.apify.com"):
        make_settings_without_env({"apify_mcp_server_url": server_url})


def test_apify_mcp_server_url_rejects_query_string_token() -> None:
    """Credentials must remain masked in SecretStr and authorization headers."""
    with pytest.raises(ValidationError, match="must not contain a token"):
        make_settings_without_env(
            {
                "apify_mcp_server_url": (
                    "https://mcp.apify.com?token=unsafe&actors=example/actor"
                )
            }
        )


@pytest.mark.parametrize(
    ("settings_data", "missing_setting"),
    [
        ({"apify_api_token": "test-token"}, "APIFY_MCP_SERVER_URL"),
        (
            {"apify_mcp_server_url": "https://mcp.apify.com"},
            "APIFY_API_TOKEN",
        ),
        (
            {
                "apify_mcp_server_url": "https://mcp.apify.com",
                "apify_api_token": "   ",
            },
            "APIFY_API_TOKEN",
        ),
    ],
)
def test_create_apify_mcp_client_requires_complete_configuration(
    settings_data: dict[str, object],
    missing_setting: str,
) -> None:
    """Missing connection values should fail before constructing a client."""
    settings = make_settings_without_env(settings_data)
    client_factory = Mock(spec=MultiServerMCPClient)

    with pytest.raises(MCPConfigurationError, match=missing_setting):
        create_apify_mcp_client(
            settings,
            client_factory=client_factory,
        )

    client_factory.assert_not_called()


def test_create_apify_mcp_client_builds_streamable_http_configuration() -> None:
    """The factory should pass explicit URL and bearer authentication settings."""
    settings = make_settings_without_env(
        {
            "apify_mcp_server_url": (
                "https://mcp.apify.com?tools=example/flipp-flyer-digest"
            ),
            "apify_api_token": "  test-token  ",
        }
    )
    expected_client = Mock(spec=MultiServerMCPClient)
    client_factory = Mock(spec=MultiServerMCPClient, return_value=expected_client)

    client = create_apify_mcp_client(
        settings,
        client_factory=client_factory,
    )

    assert client is expected_client
    client_factory.assert_called_once_with(
        {
            APIFY_FLIPP_SERVER_NAME: {
                "transport": "streamable_http",
                "url": ("https://mcp.apify.com/?tools=example/flipp-flyer-digest"),
                "headers": {"Authorization": "Bearer test-token"},
            }
        },
        handle_tool_errors=True,
    )
