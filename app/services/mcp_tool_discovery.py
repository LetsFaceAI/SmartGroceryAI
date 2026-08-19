"""Discover Apify MCP tools without coupling them to an agent or business flow.

MCP discovery performs network I/O, so this module exposes an asynchronous service
boundary. It returns standard LangChain ``BaseTool`` objects but deliberately does
not bind them to a model, create an agent, or invoke any flyer tool.
"""

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.logging import get_logger
from app.core.mcp import APIFY_FLIPP_SERVER_NAME, create_apify_mcp_client

logger = get_logger(__name__)


class MCPToolDiscoveryError(RuntimeError):
    """Report MCP discovery failures without exposing connection credentials."""


async def discover_apify_mcp_tools(
    client: MultiServerMCPClient | None = None,
) -> list[BaseTool]:
    """Connect to the configured Apify server and return its LangChain tools.

    Args:
        client: Optional configured client. Tests inject a mock; production callers
            normally omit it and reuse the centralized MCP client factory.

    Returns:
        LangChain-compatible tools advertised by the configured Apify server.

    Raises:
        MCPToolDiscoveryError: If discovery fails or the server advertises no tools.
        MCPConfigurationError: If the default client lacks required configuration.
    """
    resolved_client = client or create_apify_mcp_client()

    try:
        tools = await resolved_client.get_tools(server_name=APIFY_FLIPP_SERVER_NAME)
    except Exception as exc:
        # Exception messages from HTTP libraries can contain request details. Log
        # only the exception type so credentials can never enter application logs.
        logger.error(
            "Apify MCP tool discovery failed error_type=%s",
            type(exc).__name__,
        )
        raise MCPToolDiscoveryError(
            "Could not discover tools from the configured Apify MCP server."
        ) from exc

    if not tools:
        logger.warning("Apify MCP server returned no tools.")
        raise MCPToolDiscoveryError(
            "The configured Apify MCP server did not advertise any tools."
        )

    for tool in tools:
        # Names are useful operational context at INFO. Full descriptions and
        # schemas remain available at DEBUG without flooding normal smoke output.
        # Authentication headers and client configuration are never logged.
        logger.info("Discovered MCP tool name=%s", tool.name)
        logger.debug(
            "MCP tool details name=%s description=%s input_schema=%s",
            tool.name,
            tool.description,
            tool.args,
        )

    return tools
