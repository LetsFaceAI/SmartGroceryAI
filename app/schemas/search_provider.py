"""Provider-neutral contracts for grocery offer searches.

These schemas are the boundary between application orchestration and external
search implementations. They deliberately contain validated domain objects rather
than MCP responses, web-search documents, or other provider-specific payloads.
"""

from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from app.schemas.location import PostalCode
from app.schemas.product_offer import ProductOffer
from app.schemas.shopping import ShoppingItem


class GrocerySearchIntent(StrEnum):
    """Describe the pricing evidence a provider should search for."""

    FLYER_DEALS = "flyer_deals"
    REGULAR_PRICES = "regular_prices"


class GrocerySearchRequest(BaseModel):
    """Represent one provider-neutral search for one requested grocery item.

    ``store`` is an optional scope rather than a third pricing intent. This permits
    both "flyer deals at Store A" and "regular prices at Store A" without creating
    ambiguous combinations of intent values.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", frozen=True)

    item: ShoppingItem
    postal_code: PostalCode = Field(
        description="Canonical Canadian postal code or US ZIP code for local results."
    )
    intent: GrocerySearchIntent = Field(
        default=GrocerySearchIntent.FLYER_DEALS,
        description="Whether to seek promotional flyer deals or regular prices.",
    )
    store: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Optional store scope; when present, the search is store-specific.",
    )
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Provider-neutral cap on validated offers returned.",
    )


class SearchProvenance(BaseModel):
    """Identify which provider produced a result and when it was retrieved.

    Each ``ProductOffer.source`` retains record-level provenance. This object adds
    result-level provider context without retaining an unvalidated raw response.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", frozen=True)

    provider_name: str = Field(min_length=1, max_length=80)
    retrieved_at: AwareDatetime
    provider_request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Optional safe provider identifier used for operational tracing.",
    )


class GrocerySearchResult(BaseModel):
    """Return validated offers from any grocery search provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request: GrocerySearchRequest
    offers: tuple[ProductOffer, ...] = Field(default=())
    provenance: SearchProvenance

    @model_validator(mode="after")
    def validate_result_limit(self) -> Self:
        """Reject providers that return more offers than the shared request allows."""
        if len(self.offers) > self.request.max_results:
            raise ValueError("offers cannot exceed the request max_results limit.")
        return self
