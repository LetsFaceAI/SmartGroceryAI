"""Validated, compact grocery-offer candidates exposed to the coordinator."""

from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from app.schemas.location import PostalCode
from app.schemas.product_offer import ProductOffer


class FlyerOfferValidity(StrEnum):
    """Describe an offer's validity relative to the search date."""

    ACTIVE = "active"
    UPCOMING = "upcoming"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


def determine_offer_validity(
    offer: ProductOffer,
    *,
    as_of: date,
) -> FlyerOfferValidity:
    """Classify an offer without deciding whether it is the best candidate."""
    if offer.valid_from is not None and offer.valid_from > as_of:
        return FlyerOfferValidity.UPCOMING

    if offer.valid_until is not None and offer.valid_until < as_of:
        return FlyerOfferValidity.EXPIRED

    if offer.valid_from is None and offer.valid_until is None:
        return FlyerOfferValidity.UNKNOWN

    return FlyerOfferValidity.ACTIVE


class FlyerOfferCandidate(ProductOffer):
    """Expose one validated offer plus its date-relative validity."""

    validity_status: FlyerOfferValidity

    @classmethod
    def from_offer(
        cls,
        offer: ProductOffer,
        *,
        as_of: date,
    ) -> Self:
        """Create an LLM-safe candidate from an already validated offer."""
        return cls.model_validate(
            {
                **offer.model_dump(),
                "validity_status": determine_offer_validity(
                    offer,
                    as_of=as_of,
                ),
            }
        )


class FlyerOfferCandidateSet(BaseModel):
    """Provide unranked, validated candidates for model-guided selection."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    requested_item_name: str = Field(min_length=1, max_length=200)
    postal_code: PostalCode
    store: str | None = Field(default=None, min_length=1, max_length=120)
    as_of: date
    provider_name: str = Field(min_length=1, max_length=80)
    retrieved_at: AwareDatetime
    offers: tuple[FlyerOfferCandidate, ...] = Field(
        default=(),
        max_length=100,
    )
