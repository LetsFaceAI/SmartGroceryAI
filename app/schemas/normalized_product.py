"""Canonical product representation used by comparison-oriented services.

``ProductOffer`` remains the validated external-data boundary. ``NormalizedProduct``
is a deterministic projection of that offer with a stable product name and the
fields future matching and price-comparison code will need most often.
"""

from decimal import Decimal
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.product_offer import (
    Money,
    PackageAmount,
    ProductOffer,
    PromotionStatus,
)


class CanonicalUnit(StrEnum):
    """Units emitted by deterministic normalization.

    Metric mass and volume use their smaller common flyer units so values can be
    compared without repeatedly scaling kilograms or litres. Imperial units stay
    separate until an explicit cross-system conversion policy is introduced.
    """

    EACH = "each"
    COUNT = "count"
    PACK = "pack"
    GRAM = "g"
    MILLILITRE = "mL"
    OUNCE = "oz"
    POUND = "lb"


class NormalizedProduct(BaseModel):
    """Capture one canonical, comparison-ready view of a product offer.

    The original offer is retained for provenance and access to validity dates or
    future external fields without weakening this model's focused comparison API.
    Duplicate price and package fields are intentional: consumers should not need
    to understand the external offer boundary to perform basic comparisons.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    normalized_name: str = Field(
        min_length=1,
        max_length=200,
        description="Canonical product name used as an initial comparison key.",
    )
    brand: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Canonical brand text when the original offer supplies it.",
    )
    store: str = Field(
        min_length=1,
        max_length=120,
        description="Canonical store name for the advertised offer.",
    )
    package_quantity: int | None = Field(
        default=None,
        ge=1,
        description="Number of equal packages in a multipack; one for a single item.",
    )
    package_size: PackageAmount | None = Field(
        default=None,
        description="Canonical size of each package before multiplying quantity.",
    )
    total_package_size: PackageAmount | None = Field(
        default=None,
        description="Canonical total size used by later unit-price comparisons.",
    )
    unit: CanonicalUnit | None = Field(
        default=None,
        description="Canonical unit shared by package_size and total_package_size.",
    )
    price: Money = Field(description="Current exact advertised price.")
    regular_price: Money | None = Field(
        default=None,
        description="Exact regular price when supplied by the flyer.",
    )
    currency: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="Currency shared by current and regular prices.",
    )
    promotion_status: PromotionStatus = Field(
        description="Validated promotion classification from ProductOffer.",
    )
    original_offer: ProductOffer = Field(
        description="Validated source offer retained for provenance and traceability.",
    )

    @model_validator(mode="after")
    def validate_package_fields(self) -> Self:
        """Require package fields as one consistent group when size is available."""
        package_fields = (
            self.package_quantity,
            self.package_size,
            self.total_package_size,
            self.unit,
        )
        if all(value is None for value in package_fields):
            return self
        if any(value is None for value in package_fields):
            raise ValueError(
                "Package quantity, size, total size, and unit must be supplied together."
            )

        # The None cases were rejected above; local aliases let mypy follow that
        # invariant while Decimal arithmetic keeps the comparison exact.
        quantity = self.package_quantity
        size = self.package_size
        total_size = self.total_package_size
        assert quantity is not None and size is not None and total_size is not None
        if total_size != size * Decimal(quantity):
            raise ValueError(
                "total_package_size must equal package_size times package_quantity."
            )
        return self
