"""Typed input and raw output contracts for one cost-conscious flyer search.

The response intentionally remains opaque at this boundary. A later mapper will
interpret Actor records and validate them as ``ProductOffer`` instances without
coupling raw MCP execution to application data normalization.
"""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_RAW_SEARCH_ITEMS = 5


class RawFlyerSearchRequest(BaseModel):
    """Validate the small input accepted by the raw Flipp search service."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(
        min_length=1,
        max_length=100,
        description="Product keyword to search in current nearby flyers.",
    )
    postal_code: str = Field(
        min_length=3,
        max_length=10,
        description="US ZIP code or Canadian postal code used for local offers.",
    )
    # The Actor supports much larger values, but this service deliberately caps
    # each paid request to protect a small development account from accidental use.
    max_items: int = Field(
        default=1,
        ge=1,
        le=MAX_RAW_SEARCH_ITEMS,
        description="Maximum raw flyer records requested from one Actor run.",
    )

    @field_validator("postal_code", mode="before")
    @classmethod
    def normalize_and_validate_postal_code(cls, value: object) -> object:
        """Return a canonical location before a cost-sensitive Actor invocation.

        This runs before length validation so harmless internal whitespace cannot
        make an otherwise valid Canadian postal code exceed the field limit.
        """
        if not isinstance(value, str):
            return value

        stripped_value = value.strip()
        compact_value = re.sub(r"\s+", "", stripped_value)
        is_canadian = bool(
            re.fullmatch(r"[A-Za-z]\d[A-Za-z]\d[A-Za-z]\d", compact_value)
        )
        if is_canadian:
            uppercase_value = compact_value.upper()
            return f"{uppercase_value[:3]} {uppercase_value[3:]}"

        is_us = bool(re.fullmatch(r"\d{5}(?:-\d{4})?", stripped_value))
        if not is_canadian and not is_us:
            raise ValueError(
                "postal_code must be a Canadian postal code or US ZIP code."
            )
        return stripped_value

    def to_tool_input(self) -> dict[str, Any]:
        """Build the explicit Actor payload using its published input field names."""
        return {
            "mode": "search",
            "postalCode": self.postal_code,
            "query": self.query,
            "itemType": "flyer",
            "sortBy": "relevancy",
            "includeRelatedItems": False,
            "includeCoupons": False,
            "maxItems": self.max_items,
        }


class RawFlyerSearchResult(BaseModel):
    """Wrap an unmodified LangChain tool response with its validated request."""

    tool_name: str
    request: RawFlyerSearchRequest
    # MCP tools can return text, content blocks, or provider-specific structured
    # content. Keeping Any here preserves the adapter response without guessing.
    raw_response: Any
