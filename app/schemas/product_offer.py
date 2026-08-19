"""Normalized product-offer schema for grocery flyer data.

External flyer records are often inconsistent or incomplete. ``ProductOffer`` is
the validated boundary that future Apify/MCP adapters will produce before deal
comparison, ranking, or AI reasoning uses the data. This module intentionally has
no knowledge of any specific data provider.
"""

from datetime import date
from decimal import Decimal
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Decimal preserves exact base-10 prices. The aliases also document the precision
# accepted at this normalized boundary instead of repeating constraints per field.
Money = Annotated[
    Decimal,
    Field(ge=Decimal("0.00"), max_digits=10, decimal_places=2),
]
PackageAmount = Annotated[
    Decimal,
    Field(gt=Decimal("0"), max_digits=10, decimal_places=3),
]


class ProductOffer(BaseModel):
    """Represent one normalized product offer from a grocery flyer.

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
    unit: str | None = Field(
        default=None,
        min_length=1,
        max_length=30,
        description="Optional package unit such as L, kg, g, or count.",
    )
    # None distinguishes missing flyer metadata from an explicit regular-price offer.
    is_on_sale: bool | None = Field(
        default=None,
        description="True for a sale, false for regular price, or null when unknown.",
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

    @model_validator(mode="after")
    def validate_offer_consistency(self) -> Self:
        """Reject contradictory price and validity-window combinations."""
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until < self.valid_from
        ):
            raise ValueError("valid_until cannot be earlier than valid_from.")

        if (
            self.is_on_sale is True
            and self.regular_price is not None
            and self.price > self.regular_price
        ):
            raise ValueError(
                "A sale price cannot be greater than the supplied regular price."
            )

        return self
