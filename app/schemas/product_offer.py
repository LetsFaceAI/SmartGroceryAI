"""Validated external product-offer schema for grocery flyer data.

External flyer records are often inconsistent or incomplete. ``ProductOffer`` is
the validated ingestion boundary that Apify/MCP adapters produce before a separate
normalization layer prepares offers for comparison, ranking, or AI reasoning. This
module intentionally has no knowledge of any specific data provider.
"""

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Decimal preserves exact base-10 prices. The aliases also document the precision
# accepted at this normalized boundary instead of repeating constraints per field.
Money = Annotated[
    Decimal,
    Field(gt=Decimal("0.00"), max_digits=10, decimal_places=2),
]
PackageAmount = Annotated[
    Decimal,
    Field(gt=Decimal("0"), max_digits=10, decimal_places=3),
]
PackageQuantity = Annotated[int, Field(ge=1, le=1_000)]


class PromotionStatus(StrEnum):
    """Describe whether pricing is regular, promotional, or unavailable."""

    UNKNOWN = "unknown"
    REGULAR = "regular"
    SALE = "sale"


class MeasurementUnit(StrEnum):
    """Provide a small normalized vocabulary for common grocery package units."""

    EACH = "each"
    COUNT = "count"
    PACK = "pack"
    GRAM = "g"
    KILOGRAM = "kg"
    MILLILITRE = "mL"
    LITRE = "L"
    OUNCE = "oz"
    POUND = "lb"


class ProductOffer(BaseModel):
    """Represent one validated external product offer from a grocery flyer.

    Product name, store, current price, and source are required because an offer
    cannot be compared or traced without them. Other fields remain optional to
    reflect gaps commonly found in flyer records.
    """

    # Unknown fields are rejected so provider mapping mistakes are visible at the
    # ingestion boundary. Whitespace normalization keeps names consistent enough for
    # later matching without prematurely changing capitalization or wording.
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    product_name: str = Field(
        min_length=1,
        max_length=200,
        description="Product name shown in the flyer.",
    )
    brand: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Optional product brand when the flyer provides one.",
    )
    store: str = Field(
        min_length=1,
        max_length=120,
        description="Store offering the advertised price.",
    )
    price: Money = Field(description="Current advertised price in the given currency.")
    regular_price: Money | None = Field(
        default=None,
        description="Optional non-sale price used to measure savings.",
    )
    currency: str = Field(
        default="CAD",
        pattern=r"^[A-Z]{3}$",
        description="Three-letter ISO-style currency code; defaults to Canadian dollars.",
    )
    package_size: PackageAmount | None = Field(
        default=None,
        description="Optional numeric package amount, such as 4 for a four-litre bag.",
    )
    package_quantity: PackageQuantity = Field(
        default=1,
        description="Number of equal packages; one preserves ordinary package behavior.",
    )
    unit: MeasurementUnit | None = Field(
        default=None,
        description="Optional normalized package measurement unit.",
    )
    promotion_status: PromotionStatus = Field(
        default=PromotionStatus.UNKNOWN,
        description="Whether the offer is a sale, regular price, or unknown.",
    )
    valid_from: date | None = Field(
        default=None,
        description="Optional first calendar date on which the offer is valid.",
    )
    valid_until: date | None = Field(
        default=None,
        description="Optional final calendar date on which the offer is valid.",
    )
    source: str = Field(
        min_length=1,
        max_length=500,
        description="Provider name, record identifier, or URL used for provenance.",
    )

    @field_validator("unit", "promotion_status", mode="before")
    @classmethod
    def normalize_enum_text(cls, value: object) -> object:
        """Trim raw enum strings before matching their normalized values."""
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_offer_consistency(self) -> Self:
        """Reject contradictory price and validity-window combinations."""
        if self.package_quantity > 1 and (
            self.package_size is None or self.unit is None
        ):
            raise ValueError(
                "A multipack quantity requires both package_size and unit."
            )

        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until < self.valid_from
        ):
            raise ValueError("valid_until cannot be earlier than valid_from.")

        if self.regular_price is None:
            return self

        if (
            self.promotion_status is PromotionStatus.SALE
            and self.price >= self.regular_price
        ):
            raise ValueError(
                "A sale price must be lower than the supplied regular price."
            )

        if (
            self.promotion_status is PromotionStatus.REGULAR
            and self.price != self.regular_price
        ):
            raise ValueError(
                "Regular-status price must equal the supplied regular price."
            )

        if (
            self.promotion_status is PromotionStatus.UNKNOWN
            and self.price > self.regular_price
        ):
            raise ValueError(
                "Unknown-status price cannot exceed the supplied regular price."
            )

        return self
