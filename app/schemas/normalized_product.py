"""Canonical product representation used by comparison-oriented services.

``ProductOffer`` remains the validated external-data boundary. ``NormalizedProduct``
is a deterministic projection of that offer with a stable product name and the
fields future matching and price-comparison code will need most often.
"""

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.product_offer import (
    MeasurementUnit,
    Money,
    PackageAmount,
    ProductOffer,
    PromotionStatus,
)


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
        description="Validated brand from the original offer when available.",
    )
    store: str = Field(description="Store advertising the original offer.")
    package_size: PackageAmount | None = Field(
        default=None,
        description="Package quantity or size without unit conversion.",
    )
    unit: MeasurementUnit | None = Field(
        default=None,
        description="Existing normalized measurement unit from ProductOffer.",
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
