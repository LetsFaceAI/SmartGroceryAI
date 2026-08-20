"""Contracts for deterministic, per-offer price comparison results.

Normalization establishes product identity and package measurements. These models
add comparison readiness and exact unit-price data without performing matching,
ranking, or price calculations themselves.
"""

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.normalized_product import CanonicalUnit, NormalizedProduct
from app.schemas.product_match import ProductMatchDecision, ProductMatchResult
from app.schemas.product_offer import PackageAmount, PriceBasis, ProductOffer

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
    UNKNOWN_PRICE_BASIS = "unknown_price_basis"
    INELIGIBLE_VALIDITY = "ineligible_validity"


class OfferValidityStatus(StrEnum):
    """Classify an offer against an explicit calendar date."""

    ACTIVE = "active"
    UPCOMING = "upcoming"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


def determine_offer_validity(
    offer: ProductOffer,
    *,
    as_of: date,
) -> OfferValidityStatus:
    """Classify an inclusive flyer window without consulting the system clock."""
    if offer.valid_from is None and offer.valid_until is None:
        return OfferValidityStatus.UNKNOWN
    if offer.valid_from is not None and as_of < offer.valid_from:
        return OfferValidityStatus.UPCOMING
    if offer.valid_until is not None and as_of > offer.valid_until:
        return OfferValidityStatus.EXPIRED
    if offer.valid_from is None or offer.valid_until is None:
        return OfferValidityStatus.UNKNOWN
    return OfferValidityStatus.ACTIVE


class UnitPriceUnit(StrEnum):
    """Define the denominator displayed for a calculated unit price."""

    KILOGRAM = "kg"
    LITRE = "L"
    ITEM = "item"
    PACK = "pack"


# This table is the schema-level comparison contract. Calculation code may produce
# these pairs, but validation must also reject impossible states created by another
# caller or deserialized data.
UNIT_PRICE_UNIT_BY_COMPARABLE_UNIT: dict[CanonicalUnit, UnitPriceUnit] = {
    CanonicalUnit.GRAM: UnitPriceUnit.KILOGRAM,
    CanonicalUnit.MILLILITRE: UnitPriceUnit.LITRE,
    CanonicalUnit.EACH: UnitPriceUnit.ITEM,
    CanonicalUnit.COUNT: UnitPriceUnit.ITEM,
    CanonicalUnit.PACK: UnitPriceUnit.PACK,
}


class OfferPriceComparison(BaseModel):
    """Represent one normalized offer's readiness and exact unit price.

    ``original_offer`` remains explicit for provenance even though the normalized
    product also retains it. The duplicated reference makes the result convenient
    for callers and is validated to prevent contradictory source data.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_offer: ProductOffer
    normalized_product: NormalizedProduct
    as_of: date
    validity_status: OfferValidityStatus
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
        if self.validity_status is not determine_offer_validity(
            self.original_offer,
            as_of=self.as_of,
        ):
            raise ValueError(
                "validity_status must match the offer window and explicit as_of date."
            )
        if (
            self.status is PriceComparisonStatus.COMPARABLE
            and self.validity_status is not OfferValidityStatus.ACTIVE
        ):
            raise ValueError("A comparable offer must be active as of the stated date.")
        if (
            self.status is PriceComparisonStatus.INELIGIBLE_VALIDITY
            and self.validity_status is OfferValidityStatus.ACTIVE
        ):
            raise ValueError("An active offer cannot have ineligible validity.")
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
            comparable_unit = self.comparable_unit
            unit_price_unit = self.unit_price_unit
            assert comparable_unit is not None and unit_price_unit is not None
            expected_unit_price_unit = UNIT_PRICE_UNIT_BY_COMPARABLE_UNIT.get(
                comparable_unit
            )
            if unit_price_unit is not expected_unit_price_unit:
                raise ValueError("unit_price_unit is invalid for the comparable_unit.")
            if (
                self.original_offer.price_basis
                in {PriceBasis.TOTAL_PACKAGE, PriceBasis.EACH}
                and self.normalized_product.total_package_size is not None
                and (
                    self.comparable_quantity
                    != self.normalized_product.total_package_size
                    or self.comparable_unit is not self.normalized_product.unit
                )
            ):
                raise ValueError(
                    "Package-price comparison values must match normalized package data."
                )
            return self

        if self.status in {
            PriceComparisonStatus.MISSING_PACKAGE_DATA,
            PriceComparisonStatus.UNKNOWN_PRICE_BASIS,
            PriceComparisonStatus.INELIGIBLE_VALIDITY,
        }:
            if any(value is not None for value in comparison_values):
                raise ValueError(
                    "This non-comparable status cannot contain comparison values."
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
    uncertain_candidates: tuple[ProductMatchResult, ...] = ()

    @model_validator(mode="after")
    def validate_selection_state(self) -> Self:
        """Prevent a selection status from contradicting its ranked output."""
        if any(
            candidate.decision is not ProductMatchDecision.UNCERTAIN
            for candidate in self.uncertain_candidates
        ):
            raise ValueError(
                "uncertain_candidates must contain uncertain matches only."
            )
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
                earlier.unit_price is None
                or later.unit_price is None
                or earlier.unit_price > later.unit_price
                for earlier, later in zip(
                    self.ranked_comparable_offers,
                    self.ranked_comparable_offers[1:],
                    strict=False,
                )
            ):
                raise ValueError(
                    "Ranked offers must be ordered by nondecreasing unit_price."
                )
            minimum_price = self.ranked_comparable_offers[0].unit_price
            expected_ties = tuple(
                comparison
                for comparison in self.ranked_comparable_offers
                if comparison.unit_price == minimum_price
            )
            if self.tied_cheapest_offers != expected_ties:
                raise ValueError(
                    "tied_cheapest_offers must exactly match the minimum-price "
                    "offers in ranked order."
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
