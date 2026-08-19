"""Execute one raw Flipp Actor search without mapping or agent involvement.

This service selects the known Actor tool from MCP discovery and invokes it once.
It has no retry loop: every real invocation may consume Apify credit, so callers
must make any later retry as a separate, explicit decision.
"""

import asyncio

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.mcp import create_apify_mcp_client
from app.schemas.flyer_search import RawFlyerSearchRequest, RawFlyerSearchResult
from app.services.mcp_tool_discovery import (
    MCPToolDiscoveryError,
    discover_apify_mcp_tools,
)

FLIPP_FLYER_TOOL_NAME = "crawlerbros--flipp-grocery-deals-scraper"
MAX_TOOL_TIMEOUT_SECONDS = 600.0

logger = get_logger(__name__)


class RawFlyerSearchError(RuntimeError):
    """Report a failed raw search without leaking credentials or request headers."""


class RawFlyerSearchTimeoutError(RawFlyerSearchError):
    """Report that the single MCP operation exceeded its client-side deadline."""


def _find_flyer_tool(tools: list[BaseTool]) -> BaseTool:
    """Select the configured Actor tool by its stable MCP-exposed name."""
    for tool in tools:
        if tool.name == FLIPP_FLYER_TOOL_NAME:
            return tool
    raise RawFlyerSearchError(
        f"Required MCP tool '{FLIPP_FLYER_TOOL_NAME}' was not discovered. "
        "Check APIFY_MCP_SERVER_URL and its tools selection."
    )


async def search_raw_flyer_offers(
    request: RawFlyerSearchRequest,
    *,
    client: MultiServerMCPClient | None = None,
    timeout_seconds: float | None = None,
) -> RawFlyerSearchResult:
    """Discover and invoke the Flipp tool exactly once with a bounded input.

    Args:
        request: Validated query, postal code, and small result limit.
        client: Optional client injection used by offline tests.
        timeout_seconds: Optional total deadline for discovery plus execution.

    Returns:
        The untouched LangChain tool response and its originating request.

    Raises:
        ValueError: If an explicit timeout falls outside the supported range.
        RawFlyerSearchTimeoutError: If discovery or execution exceeds the deadline.
        RawFlyerSearchError: If discovery, tool selection, or invocation fails.
    """
    resolved_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else get_settings().apify_mcp_tool_timeout_seconds
    )
    if not 0 < resolved_timeout <= MAX_TOOL_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout_seconds must be greater than 0 and no more than "
            f"{MAX_TOOL_TIMEOUT_SECONDS:g}."
        )

    # Raw execution needs MCP isError responses to raise instead of being converted
    # into model-readable content that could be mistaken for successful raw data.
    resolved_client = client or create_apify_mcp_client(handle_tool_errors=False)

    try:
        # The one deadline includes discovery and the one paid Actor invocation.
        # asyncio.timeout cancels waiting locally; it does not trigger a retry.
        async with asyncio.timeout(resolved_timeout):
            tools = await discover_apify_mcp_tools(resolved_client)
            flyer_tool = _find_flyer_tool(tools)
            raw_response = await flyer_tool.ainvoke(request.to_tool_input())
    except TimeoutError as exc:
        logger.error(
            "Raw flyer search timed out tool=%s timeout_seconds=%s",
            FLIPP_FLYER_TOOL_NAME,
            resolved_timeout,
        )
        raise RawFlyerSearchTimeoutError(
            "The raw flyer search timed out and was not retried. The Apify Actor "
            "may still be running; check Apify Console before trying again."
        ) from exc
    except RawFlyerSearchError:
        raise
    except MCPToolDiscoveryError as exc:
        raise RawFlyerSearchError(
            "Could not discover the configured Flipp MCP tool."
        ) from exc
    except Exception as exc:
        # Avoid exception messages because HTTP failures can include sensitive
        # request details. Logging the type is sufficient for safe diagnostics.
        logger.error(
            "Raw flyer search failed tool=%s error_type=%s",
            FLIPP_FLYER_TOOL_NAME,
            type(exc).__name__,
        )
        raise RawFlyerSearchError(
            "The configured Flipp MCP tool failed during its single invocation."
        ) from exc

    logger.info(
        "Raw flyer search completed tool=%s max_items=%s",
        FLIPP_FLYER_TOOL_NAME,
        request.max_items,
    )
    return RawFlyerSearchResult(
        tool_name=FLIPP_FLYER_TOOL_NAME,
        request=request,
        raw_response=raw_response,
    )
