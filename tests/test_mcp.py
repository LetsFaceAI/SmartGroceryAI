"""Unit tests for offline Apify MCP client configuration."""

import logging
from unittest.mock import Mock, patch

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


@pytest.mark.parametrize(
    "parameter",
    ["token", "api_token", "access_token", "api-key", "unknown"],
)
def test_apify_mcp_server_url_rejects_undocumented_query_parameters(
    parameter: str,
) -> None:
    """An allowlist blocks credential aliases and unknown forwarding behavior."""
    with pytest.raises(ValidationError, match="unsupported query parameter"):
        make_settings_without_env(
            {
                "apify_mcp_server_url": (
                    f"https://mcp.apify.com?{parameter}=unsafe&actors=example/actor"
                )
            }
        )


@pytest.mark.parametrize(
    "query",
    [
        "actors=example/actor",
        "tools=actors&ui=false&telemetry-enabled=false",
    ],
)
def test_apify_mcp_server_url_accepts_documented_query_parameters(
    query: str,
) -> None:
    """Deployment may use only Apify's documented hosted-MCP controls."""
    settings = make_settings_without_env(
        {"apify_mcp_server_url": f"https://mcp.apify.com?{query}"}
    )

    assert settings.apify_mcp_server_url is not None


def test_apify_mcp_server_url_rejects_user_info_credentials() -> None:
    """URL authority fields must not provide another path for secrets."""
    with pytest.raises(ValidationError, match="must not contain credentials"):
        make_settings_without_env(
            {"apify_mcp_server_url": "https://user:secret@mcp.apify.com"}
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


def test_create_client_sanitizes_settings_validation_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invalid configuration must not expose Pydantic's rejected input value."""
    secret = "must-not-leak"
    try:
        make_settings_without_env(
            {"apify_mcp_server_url": (f"https://mcp.apify.com?access_token={secret}")}
        )
    except ValidationError as validation_error:
        settings_error = validation_error
    else:  # pragma: no cover - protects the test setup itself
        pytest.fail("Expected invalid fixture configuration.")

    client_factory = Mock(spec=MultiServerMCPClient)
    with (
        patch("app.core.mcp.get_settings", side_effect=settings_error),
        caplog.at_level(logging.ERROR),
        pytest.raises(MCPConfigurationError) as error,
    ):
        create_apify_mcp_client(client_factory=client_factory)

    public_output = f"{error.value}\n{caplog.text}"
    assert "MCP configuration is invalid" in str(error.value)
    assert secret not in public_output
    assert "access_token" not in public_output
    assert error.value.__cause__ is None
    client_factory.assert_not_called()
