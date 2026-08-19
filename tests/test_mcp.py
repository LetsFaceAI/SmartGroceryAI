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


def test_apify_mcp_server_url_requires_a_valid_http_url() -> None:
    """Malformed endpoint configuration should fail during settings validation."""
    with pytest.raises(ValidationError, match="apify_mcp_server_url"):
        Settings.model_validate({"apify_mcp_server_url": "not-a-url"})


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
    settings = Settings.model_validate(settings_data)
    client_factory = Mock(spec=MultiServerMCPClient)

    with pytest.raises(MCPConfigurationError, match=missing_setting):
        create_apify_mcp_client(
            settings,
            client_factory=client_factory,
        )

    client_factory.assert_not_called()


def test_create_apify_mcp_client_builds_streamable_http_configuration() -> None:
    """The factory should pass explicit URL and bearer authentication settings."""
    settings = Settings.model_validate(
        {
            "apify_mcp_server_url": (
                "https://mcp.apify.com?tools=example/flipp-flyer-digest"
            ),
            "apify_api_token": "test-token",
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
        }
    )
