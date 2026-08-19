"""Manually verify Apify MCP connectivity and print discovered tool names.

This script intentionally performs live network discovery and must only be run by a
developer who has configured ``APIFY_MCP_SERVER_URL`` and ``APIFY_API_TOKEN`` in
their local, untracked ``.env`` file. It never prints the token or invokes a tool.
"""

import asyncio

from app.core.logging import configure_logging
from app.core.mcp import MCPConfigurationError
from app.services.mcp_tool_discovery import (
    MCPToolDiscoveryError,
    discover_apify_mcp_tools,
)


async def main() -> None:
    """Discover live tools and display only their non-sensitive names."""
    configure_logging()

    try:
        tools = await discover_apify_mcp_tools()
    except (MCPConfigurationError, MCPToolDiscoveryError) as exc:
        # Both public error types contain safe guidance and never include tokens.
        raise SystemExit(f"MCP discovery failed: {exc}") from None

    print(f"MCP connection succeeded; discovered {len(tools)} tool(s):")
    for tool in tools:
        print(f"- {tool.name}")


if __name__ == "__main__":
    asyncio.run(main())
