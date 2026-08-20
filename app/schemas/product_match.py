"""Structured result contract for deterministic product-name matching."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> Self:
        """Prevent contradictory match flags, types, and confidence values."""
        if self.matched and self.match_type is ProductMatchType.NONE:
            raise ValueError("A matched result must identify a successful match type.")
        if not self.matched and self.match_type is not ProductMatchType.NONE:
            raise ValueError("An unmatched result must use match type 'none'.")
        if self.matched and self.confidence <= 0:
            raise ValueError("A matched result must have positive confidence.")
        if not self.matched and self.confidence != 0:
            raise ValueError("An unmatched result must have zero confidence.")
        return self
