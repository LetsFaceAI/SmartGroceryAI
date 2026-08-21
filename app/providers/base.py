"""Structural interface implemented by grocery search providers.

The protocol keeps callers dependent on an application-owned capability rather
than Apify MCP tools or Tavily APIs. Concrete providers can therefore be replaced
or mocked without changing future orchestration and agent code.
"""

from typing import Protocol, runtime_checkable

from app.schemas.search_provider import GrocerySearchRequest, GrocerySearchResult


class SearchProviderError(RuntimeError):
    """Base error exposed by provider implementations to application callers."""


class UnsupportedSearchIntentError(SearchProviderError):
    """Report that a provider cannot satisfy the requested kind of search."""


class SearchProviderExecutionError(SearchProviderError):
    """Report a provider failure without exposing transport or credential details."""


class SearchProviderTimeoutError(SearchProviderExecutionError):
    """Report a bounded provider operation that did not finish before its deadline."""


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
