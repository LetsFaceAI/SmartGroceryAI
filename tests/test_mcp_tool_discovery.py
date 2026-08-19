"""Offline unit tests for the MCP tool-discovery service."""

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.mcp import APIFY_FLIPP_SERVER_NAME
from app.services.mcp_tool_discovery import (
    MCPToolDiscoveryError,
    discover_apify_mcp_tools,
)


def make_tool(
    name: str = "search_flipp_flyers",
    description: str = "Search current grocery flyer offers.",
) -> Mock:
    """Create a BaseTool-shaped mock without an MCP server or model."""
    tool = Mock(spec=BaseTool)
    tool.name = name
    tool.description = description
    tool.args = {"postal_code": {"type": "string"}}
    return tool


@pytest.mark.anyio
async def test_discover_apify_mcp_tools_returns_and_logs_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Discovery should return adapter tools and log only public metadata."""
    tool = make_tool()
    client = Mock(spec=MultiServerMCPClient)
    client.get_tools = AsyncMock(return_value=[tool])

    with caplog.at_level(logging.DEBUG):
        discovered = await discover_apify_mcp_tools(client)

    assert discovered == [tool]
    client.get_tools.assert_awaited_once_with(server_name=APIFY_FLIPP_SERVER_NAME)
    assert "search_flipp_flyers" in caplog.text
    assert "Search current grocery flyer offers" in caplog.text
    assert "postal_code" in caplog.text


@pytest.mark.anyio
async def test_discover_apify_mcp_tools_uses_central_client_factory() -> None:
    """Omitting a client should reuse the connection factory from app.core.mcp."""
    client = Mock(spec=MultiServerMCPClient)
    client.get_tools = AsyncMock(return_value=[make_tool()])

    with patch(
        "app.services.mcp_tool_discovery.create_apify_mcp_client",
        return_value=client,
    ) as client_factory:
        await discover_apify_mcp_tools()

    client_factory.assert_called_once_with()


@pytest.mark.anyio
async def test_discover_apify_mcp_tools_rejects_empty_tool_list() -> None:
    """An empty server response should not look like successful discovery."""
    client = Mock(spec=MultiServerMCPClient)
    client.get_tools = AsyncMock(return_value=[])

    with pytest.raises(MCPToolDiscoveryError, match="did not advertise any tools"):
        await discover_apify_mcp_tools(client)


@pytest.mark.anyio
async def test_discovery_failure_does_not_log_exception_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transport errors should be wrapped without logging sensitive details."""
    client = Mock(spec=MultiServerMCPClient)
    client.get_tools = AsyncMock(
        side_effect=RuntimeError("Authorization: Bearer test-secret")
    )

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(MCPToolDiscoveryError, match="Could not discover tools"),
    ):
        await discover_apify_mcp_tools(client)

    assert "RuntimeError" in caplog.text
    assert "test-secret" not in caplog.text
