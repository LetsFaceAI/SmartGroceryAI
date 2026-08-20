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
UNIT_PRICE_DECIMAL_PLACES = 12
UnitPrice = Annotated[
    Decimal,
    Field(
        gt=Decimal("0"),
        max_digits=28,
        decimal_places=UNIT_PRICE_DECIMAL_PLACES,
    ),
]


class PriceComparisonStatus(StrEnum):
    """Describe whether and why one offer can participate in comparison."""

    COMPARABLE = "comparable"
    MISSING_PACKAGE_DATA = "missing_package_data"
    UNSUPPORTED_UNIT = "unsupported_unit"


class UnitPriceUnit(StrEnum):
    """Define the denominator displayed for a calculated unit price."""

    KILOGRAM = "kg"
    LITRE = "L"
    ITEM = "item"
    PACK = "pack"


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
        description="Exact offer price per one unit_price_unit.",
    )
    unit_price_unit: UnitPriceUnit | None = Field(
        default=None,
        description="Standard denominator used to display and rank unit_price.",
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
        if (
            self.comparable_quantity is not None
            and self.comparable_quantity != self.normalized_product.total_package_size
        ):
            raise ValueError(
                "comparable_quantity must match the normalized total package size."
            )
        if (
            self.comparable_unit is not None
            and self.comparable_unit is not self.normalized_product.unit
        ):
            raise ValueError("comparable_unit must match the normalized product unit.")

        comparison_values = (
            self.comparable_quantity,
            self.comparable_unit,
            self.unit_price,
            self.unit_price_unit,
        )
        if self.status is PriceComparisonStatus.COMPARABLE:
            if any(value is None for value in comparison_values):
                raise ValueError(
                    "Comparable offers require quantity, units, and unit_price."
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
        if self.unit_price_unit is not None:
            raise ValueError("Unsupported-unit results cannot contain unit_price_unit.")
        return self


class CheapestOfferStatus(StrEnum):
    """Describe the outcome of selecting from matched offer candidates."""

    SELECTED = "selected"
    NO_COMPARABLE_OFFERS = "no_comparable_offers"
    INCOMPATIBLE_COMPARISON_GROUPS = "incompatible_comparison_groups"


class CheapestOfferSelection(BaseModel):
    """Return deterministic ranking context without hiding rejected candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_item_name: str = Field(min_length=1, max_length=200)
    status: CheapestOfferStatus
    reason: str = Field(min_length=1, max_length=300)
    comparisons: tuple[OfferPriceComparison, ...]
    ranked_comparable_offers: tuple[OfferPriceComparison, ...] = ()
    cheapest_offer: OfferPriceComparison | None = None
    tied_cheapest_offers: tuple[OfferPriceComparison, ...] = ()
    ignored_non_matches: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_selection_state(self) -> Self:
        """Prevent a selection status from contradicting its ranked output."""
        if self.status is CheapestOfferStatus.SELECTED:
            if not self.ranked_comparable_offers or self.cheapest_offer is None:
                raise ValueError(
                    "A selected result requires ranked and cheapest offers."
                )
            if self.cheapest_offer != self.ranked_comparable_offers[0]:
                raise ValueError("cheapest_offer must be the first ranked offer.")
            if not self.tied_cheapest_offers:
                raise ValueError(
                    "A selected result must include its cheapest tie group."
                )
            if any(
                comparison.status is not PriceComparisonStatus.COMPARABLE
                for comparison in self.ranked_comparable_offers
            ):
                raise ValueError("Ranked offers must all be comparable.")
            if any(
                comparison not in self.comparisons
                for comparison in self.ranked_comparable_offers
            ):
                raise ValueError("Ranked offers must belong to comparisons.")
            if any(
                comparison not in self.ranked_comparable_offers
                for comparison in self.tied_cheapest_offers
            ):
                raise ValueError("Tied offers must belong to the ranked offers.")
            if any(
                comparison.unit_price != self.cheapest_offer.unit_price
                for comparison in self.tied_cheapest_offers
            ):
                raise ValueError(
                    "Tied offers must share the cheapest offer's unit price."
                )
            return self

        if (
            self.ranked_comparable_offers
            or self.cheapest_offer is not None
            or self.tied_cheapest_offers
        ):
            raise ValueError(
                "A non-selected result cannot contain ranked or cheapest offers."
            )
        return self
