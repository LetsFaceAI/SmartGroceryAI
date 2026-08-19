"""Offline tests for one cost-bounded raw MCP flyer search."""

import asyncio
import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import ValidationError

from app.schemas.flyer_search import RawFlyerSearchRequest, RawFlyerSearchResult
from app.services.raw_flyer_search import (
    FLIPP_FLYER_TOOL_NAME,
    RawFlyerSearchError,
    RawFlyerSearchTimeoutError,
    search_raw_flyer_offers,
)


def make_client() -> Mock:
    """Return a client-shaped mock; it never creates an MCP session."""
    return Mock(spec=MultiServerMCPClient)


def make_flyer_tool(*, response: object) -> Mock:
    """Return an async LangChain tool mock with a controlled response."""
    tool = Mock(spec=BaseTool)
    tool.name = FLIPP_FLYER_TOOL_NAME
    tool.ainvoke = AsyncMock(return_value=response)
    return tool


@pytest.mark.anyio
async def test_search_raw_flyer_offers_invokes_expected_tool_once() -> None:
    """One request should create one explicit, minimum-size Actor payload."""
    request = RawFlyerSearchRequest(
        query="milk",
        postal_code="M5V 3A8",
        max_items=1,
    )
    raw_response = [{"type": "text", "text": "sample raw response"}]
    tool = make_flyer_tool(response=raw_response)

    with patch(
        "app.services.raw_flyer_search.discover_apify_mcp_tools",
        new=AsyncMock(return_value=[tool]),
    ) as discovery:
        result = await search_raw_flyer_offers(
            request,
            client=make_client(),
            timeout_seconds=1,
        )

    assert isinstance(result, RawFlyerSearchResult)
    assert result.raw_response is raw_response
    discovery.assert_awaited_once()
    tool_call = tool.ainvoke.await_args.args[0]
    assert tool_call["name"] == FLIPP_FLYER_TOOL_NAME
    assert isinstance(tool_call["id"], str)
    assert tool_call["args"] == {
        "mode": "search",
        "postalCode": "M5V 3A8",
        "query": "milk",
        "itemType": "flyer",
        "sortBy": "relevancy",
        "includeRelatedItems": False,
        "includeCoupons": False,
        "maxItems": 1,
    }


@pytest.mark.anyio
async def test_search_raw_flyer_offers_configures_errors_to_raise() -> None:
    """The default client should surface MCP execution failures as exceptions."""
    raw_response = "sample"
    tool = make_flyer_tool(response=raw_response)
    client = make_client()

    with (
        patch(
            "app.services.raw_flyer_search.create_apify_mcp_client",
            return_value=client,
        ) as client_factory,
        patch(
            "app.services.raw_flyer_search.discover_apify_mcp_tools",
            new=AsyncMock(return_value=[tool]),
        ),
    ):
        await search_raw_flyer_offers(
            RawFlyerSearchRequest(query="milk", postal_code="10001"),
            timeout_seconds=1,
        )

    client_factory.assert_called_once_with(handle_tool_errors=False)


def test_raw_flyer_search_request_enforces_small_result_limit() -> None:
    """Application validation should reject expensive result sizes deterministically."""
    with pytest.raises(ValidationError, match="max_items"):
        RawFlyerSearchRequest(
            query="milk",
            postal_code="M5V 3A8",
            max_items=6,
        )


@pytest.mark.anyio
async def test_search_raw_flyer_offers_requires_actor_tool() -> None:
    """Supporting Apify tools must not be mistaken for the paid flyer tool."""
    other_tool = Mock(spec=BaseTool)
    other_tool.name = "get-dataset-items"

    with (
        patch(
            "app.services.raw_flyer_search.discover_apify_mcp_tools",
            new=AsyncMock(return_value=[other_tool]),
        ),
        pytest.raises(RawFlyerSearchError, match=FLIPP_FLYER_TOOL_NAME),
    ):
        await search_raw_flyer_offers(
            RawFlyerSearchRequest(query="milk", postal_code="10001"),
            client=make_client(),
            timeout_seconds=1,
        )


@pytest.mark.anyio
async def test_search_raw_flyer_offers_times_out_without_retrying() -> None:
    """A timeout should cancel local waiting and never issue a second invocation."""
    tool = make_flyer_tool(response="unused")

    async def wait_too_long(_: object) -> None:
        await asyncio.sleep(1)

    tool.ainvoke.side_effect = wait_too_long

    with (
        patch(
            "app.services.raw_flyer_search.discover_apify_mcp_tools",
            new=AsyncMock(return_value=[tool]),
        ),
        pytest.raises(RawFlyerSearchTimeoutError, match="not retried"),
    ):
        await search_raw_flyer_offers(
            RawFlyerSearchRequest(query="milk", postal_code="10001"),
            client=make_client(),
            timeout_seconds=0.01,
        )

    assert tool.ainvoke.await_count == 1


@pytest.mark.anyio
async def test_search_failure_does_not_log_exception_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invocation errors should expose their type but not sensitive details."""
    tool = make_flyer_tool(response="unused")
    tool.ainvoke.side_effect = RuntimeError("Bearer test-secret")

    with (
        patch(
            "app.services.raw_flyer_search.discover_apify_mcp_tools",
            new=AsyncMock(return_value=[tool]),
        ),
        caplog.at_level(logging.ERROR),
        pytest.raises(RawFlyerSearchError, match="single invocation"),
    ):
        await search_raw_flyer_offers(
            RawFlyerSearchRequest(query="milk", postal_code="10001"),
            client=make_client(),
            timeout_seconds=1,
        )

    assert tool.ainvoke.await_count == 1
    assert "RuntimeError" in caplog.text
    assert "test-secret" not in caplog.text
