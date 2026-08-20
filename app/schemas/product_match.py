"""Structured result contract for deterministic product-name matching."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.normalized_product import NormalizedProduct


class ProductMatchType(StrEnum):
    """Identify the deterministic rule responsible for a match decision."""

    EXACT = "exact"
    CONTAINMENT = "containment"
    TOKEN_OVERLAP = "token_overlap"
    NONE = "none"


class ProductMatchResult(BaseModel):
    """Describe whether one requested item matches one normalized flyer product."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    matched: bool
    match_type: ProductMatchType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)
    normalized_request_name: str = Field(min_length=1, max_length=200)
    product: NormalizedProduct
