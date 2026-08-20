"""Contracts for deterministic, per-offer price comparison results.

Normalization establishes product identity and package measurements. These models
add comparison readiness and exact unit-price data without performing matching,
ranking, or price calculations themselves.
"""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.normalized_product import CanonicalUnit, NormalizedProduct
from app.schemas.product_offer import PackageAmount, ProductOffer

# Unit prices often need more fractional digits than retail totals. Decimal avoids
# binary floating-point errors, while twelve decimal places provide deterministic
# precision for small per-gram or per-millilitre amounts.
UnitPrice = Annotated[
    Decimal,
    Field(gt=Decimal("0"), max_digits=20, decimal_places=12),
]


class PriceComparisonStatus(StrEnum):
    """Describe whether and why one offer can participate in comparison."""

    COMPARABLE = "comparable"
    MISSING_PACKAGE_DATA = "missing_package_data"
    UNSUPPORTED_UNIT = "unsupported_unit"


class OfferPriceComparison(BaseModel):
    """Represent one normalized offer's readiness and exact unit price.

    ``original_offer`` remains explicit for provenance even though the normalized
    product also retains it. The duplicated reference makes the result convenient
    for callers and is validated to prevent contradictory source data.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_offer: ProductOffer
    normalized_product: NormalizedProduct
    status: PriceComparisonStatus
    reason: str = Field(min_length=1, max_length=300)
    comparable_quantity: PackageAmount | None = Field(
        default=None,
        description="Canonical total quantity used as the unit-price denominator.",
    )
    comparable_unit: CanonicalUnit | None = Field(
        default=None,
        description="Canonical unit associated with comparable_quantity.",
    )
    unit_price: UnitPrice | None = Field(
        default=None,
        description="Exact offer price per one comparable unit.",
    )
    currency: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="Currency of unit_price and the original offer.",
    )

    @model_validator(mode="after")
    def validate_comparison_state(self) -> Self:
        """Keep status and optional comparison values internally consistent."""
        if self.original_offer != self.normalized_product.original_offer:
            raise ValueError(
                "original_offer must match the normalized product's source offer."
            )
        if self.currency != self.original_offer.currency:
            raise ValueError("currency must match the original offer currency.")

        comparison_values = (
            self.comparable_quantity,
            self.comparable_unit,
            self.unit_price,
        )
        if self.status is PriceComparisonStatus.COMPARABLE:
            if any(value is None for value in comparison_values):
                raise ValueError(
                    "Comparable offers require quantity, unit, and unit_price."
                )
            return self

        if self.status is PriceComparisonStatus.MISSING_PACKAGE_DATA:
            if any(value is not None for value in comparison_values):
                raise ValueError(
                    "Missing-package results cannot contain comparison values."
                )
            return self

        # An unsupported unit can still retain its known canonical measurement,
        # but no unit price is published because consumers could compare it with
        # an incompatible measurement family.
        if self.comparable_quantity is None or self.comparable_unit is None:
            raise ValueError(
                "Unsupported-unit results must retain their quantity and unit."
            )
        if self.unit_price is not None:
            raise ValueError("Unsupported-unit results cannot contain unit_price.")
        return self
