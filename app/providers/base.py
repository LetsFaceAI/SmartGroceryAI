"""Structural interface implemented by grocery search providers.

The protocol keeps callers dependent on an application-owned capability rather
than Apify MCP tools or Tavily APIs. Concrete providers can therefore be replaced
or mocked without changing future orchestration and agent code.
"""

from typing import Protocol, runtime_checkable

from app.schemas.search_provider import GrocerySearchRequest, GrocerySearchResult


@runtime_checkable
class GrocerySearchProvider(Protocol):
    """Search for validated grocery offers through one configured provider."""

    @property
    def provider_name(self) -> str:
        """Return the stable, non-secret provider identifier used in provenance."""
        ...

    async def search(self, request: GrocerySearchRequest) -> GrocerySearchResult:
        """Execute one bounded search and return only shared validated models."""
        ...
