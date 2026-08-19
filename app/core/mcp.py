"""Create the LangChain MCP client without discovering or invoking tools.

This module owns transport and authentication setup for the hosted Apify MCP
server. Business services should receive or use the resulting client rather than
assembling connection dictionaries, which keeps external-service details at one
small, mockable boundary.
"""

from collections.abc import Callable

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection

from app.core.config import Settings, get_settings

APIFY_FLIPP_SERVER_NAME = "apify_flipp"

# Accepting the constructor as a dependency lets tests inspect connection setup
# without opening a socket or depending on adapter internals.
MCPClientFactory = Callable[[dict[str, Connection]], MultiServerMCPClient]


class MCPConfigurationError(RuntimeError):
    """Report incomplete MCP configuration before any connection is attempted."""


def create_apify_mcp_client(
    settings: Settings | None = None,
    *,
    client_factory: MCPClientFactory = MultiServerMCPClient,
) -> MultiServerMCPClient:
    """Build a LangChain client configured for the hosted Apify MCP server.

    Constructing ``MultiServerMCPClient`` only stores connection settings. A later
    service will explicitly discover tools or open a session when it is ready to
    perform network I/O.

    Args:
        settings: Optional validated settings, primarily useful in tests.
        client_factory: Injectable constructor used to keep unit tests offline.

    Returns:
        A configured client whose sole server uses Streamable HTTP.

    Raises:
        MCPConfigurationError: If the server URL or Apify token is missing.
    """
    resolved_settings = settings or get_settings()
    server_url = resolved_settings.apify_mcp_server_url
    api_token = resolved_settings.apify_api_token

    if server_url is None:
        raise MCPConfigurationError(
            "APIFY_MCP_SERVER_URL is not configured. Add the Apify MCP URL for "
            "the selected Flipp Actor to your .env file."
        )

    if api_token is None or not api_token.get_secret_value().strip():
        raise MCPConfigurationError(
            "APIFY_API_TOKEN is not configured. Add an Apify API token to your "
            ".env file before creating the MCP client."
        )

    connection: Connection = {
        "transport": "streamable_http",
        "url": str(server_url),
        "headers": {
            "Authorization": f"Bearer {api_token.get_secret_value()}",
        },
    }
    return client_factory({APIFY_FLIPP_SERVER_NAME: connection})
