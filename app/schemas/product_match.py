"""Structured result contract for deterministic product-name matching."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.normalized_product import NormalizedProduct
from app.schemas.shopping import Preference, ShoppingConstraint, ShoppingItem


class ProductMatchType(StrEnum):
    """Identify the deterministic rule responsible for a match decision."""

    EXACT = "exact"
    CONTAINMENT = "containment"
    TOKEN_OVERLAP = "token_overlap"
    NONE = "none"


class ProductMatchDecision(StrEnum):
    """Separate trusted equivalence from candidates that need confirmation."""

    SAFE = "safe"
    UNCERTAIN = "uncertain"
    NO_MATCH = "no_match"


class ProductMatchRequest(BaseModel):
    """Carry the full item context into deterministic product matching.

    Notes and request-wide preferences are retained even when the deterministic
    matcher does not interpret them. This prevents later orchestration code from
    losing user intent while keeping only explicit constraints enforcement-ready.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", frozen=True)

    shopping_item: ShoppingItem
    effective_constraints: tuple[ShoppingConstraint, ...] = Field(
        default=(),
        max_length=20,
        description="Explicit and safely derived constraints used by matching.",
    )
    request_preferences: tuple[Preference, ...] = Field(default=(), max_length=20)

    @property
    def item_name(self) -> str:
        """Expose the preserved item's name for matching-oriented consumers."""
        return self.shopping_item.name

    @property
    def notes(self) -> str | None:
        """Expose original notes without copying or rewriting their contents."""
        return self.shopping_item.notes

    @property
    def constraints(self) -> tuple[ShoppingConstraint, ...]:
        """Expose the deterministic constraint set used for this match request."""
        return self.effective_constraints


class ProductMatchResult(BaseModel):
    """Describe whether one requested item matches one normalized flyer product."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    matched: bool
    candidate: bool
    safe_for_ranking: bool
    decision: ProductMatchDecision
    match_type: ProductMatchType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)
    normalized_request_name: str = Field(min_length=1, max_length=200)
    request: ProductMatchRequest
    unmet_required_constraints: tuple[str, ...] = ()
    unmet_optional_constraints: tuple[str, ...] = ()
    product: NormalizedProduct

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> Self:
        """Prevent contradictory match flags, types, and confidence values."""
        if self.decision is ProductMatchDecision.SAFE:
            if not self.matched or not self.candidate or not self.safe_for_ranking:
                raise ValueError(
                    "A safe match must be matched, retained, and ranking eligible."
                )
            if self.match_type is ProductMatchType.NONE or self.confidence <= 0:
                raise ValueError("A safe match requires a positive match rule.")
            if self.unmet_required_constraints:
                raise ValueError("A safe match cannot miss a required constraint.")
            return self

        if self.decision is ProductMatchDecision.UNCERTAIN:
            if self.matched or not self.candidate or self.safe_for_ranking:
                raise ValueError(
                    "An uncertain result must remain a non-ranking candidate."
                )
            if self.match_type is ProductMatchType.NONE or self.confidence <= 0:
                raise ValueError(
                    "An uncertain candidate requires a possible match rule."
                )
            return self

        if self.matched or self.candidate or self.safe_for_ranking:
            raise ValueError("A no-match result cannot be retained or ranked.")
        if self.match_type is not ProductMatchType.NONE or self.confidence != 0:
            raise ValueError(
                "A no-match result must use type 'none' and zero confidence."
            )
        return self
