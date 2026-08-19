"""Map simple external flyer records into validated ``ProductOffer`` objects.

The raw keys below are intentionally provider-neutral placeholders. When the Apify
MCP response shape is known, this module can change its key and value normalization
without forcing comparison, planning, or AI code to understand that external shape.
"""

from collections.abc import Mapping
from typing import cast

from pydantic import ValidationError

from app.schemas.product_offer import (
    MeasurementUnit,
    ProductOffer,
    PromotionStatus,
)

# Field names are mapped explicitly instead of unpacking arbitrary external data into
# Pydantic. This documents the accepted boundary and safely ignores provider metadata
# that the application does not yet use.
RAW_FIELD_MAP: dict[str, str] = {
    "product": "product_name",
    "brand": "brand",
    "store": "store",
    "price": "price",
    "regularPrice": "regular_price",
    "currency": "currency",
    "packageSize": "package_size",
    "unit": "unit",
    "validFrom": "valid_from",
    "validUntil": "valid_until",
    "source": "source",
}

# External sources commonly spell the same unit in several ways. Keep aliases here,
# not in the core schema, because normalization is an ingestion responsibility.
UNIT_ALIASES: dict[str, MeasurementUnit] = {
    "ea": MeasurementUnit.EACH,
    "each": MeasurementUnit.EACH,
    "ct": MeasurementUnit.COUNT,
    "count": MeasurementUnit.COUNT,
    "pk": MeasurementUnit.PACK,
    "pack": MeasurementUnit.PACK,
    "g": MeasurementUnit.GRAM,
    "gram": MeasurementUnit.GRAM,
    "grams": MeasurementUnit.GRAM,
    "kg": MeasurementUnit.KILOGRAM,
    "kilogram": MeasurementUnit.KILOGRAM,
    "kilograms": MeasurementUnit.KILOGRAM,
    "ml": MeasurementUnit.MILLILITRE,
    "millilitre": MeasurementUnit.MILLILITRE,
    "millilitres": MeasurementUnit.MILLILITRE,
    "l": MeasurementUnit.LITRE,
    "litre": MeasurementUnit.LITRE,
    "litres": MeasurementUnit.LITRE,
    "liter": MeasurementUnit.LITRE,
    "liters": MeasurementUnit.LITRE,
    "oz": MeasurementUnit.OUNCE,
    "ounce": MeasurementUnit.OUNCE,
    "ounces": MeasurementUnit.OUNCE,
    "lb": MeasurementUnit.POUND,
    "pound": MeasurementUnit.POUND,
    "pounds": MeasurementUnit.POUND,
}


class ProductOfferMappingError(ValueError):
    """Report raw flyer data that cannot become a valid product offer."""


def _normalize_unit(value: object) -> object:
    """Translate a known unit alias while leaving invalid values for validation."""
    if isinstance(value, str):
        normalized_value = value.strip()
        return UNIT_ALIASES.get(normalized_value.casefold(), normalized_value)
    return value


def _map_promotion_status(raw_offer: Mapping[str, object]) -> PromotionStatus | None:
    """Translate an optional external sale flag into the normalized status enum."""
    if "onSale" not in raw_offer or raw_offer["onSale"] is None:
        return None

    sale_flag = raw_offer["onSale"]
    if not isinstance(sale_flag, bool):
        raise ProductOfferMappingError(
            "Raw field 'onSale' must be true, false, or null."
        )

    return PromotionStatus.SALE if sale_flag else PromotionStatus.REGULAR


def map_product_offer(raw_offer: object) -> ProductOffer:
    """Convert a raw external flyer mapping into a validated product offer.

    Args:
        raw_offer: External data expected to be a dictionary-like mapping using the
            keys in ``RAW_FIELD_MAP``.

    Returns:
        A normalized and validated ``ProductOffer``.

    Raises:
        ProductOfferMappingError: If required fields are missing, values fail schema
            validation, the top-level value is not a mapping, or a supported raw
            field has an invalid representation.
    """
    if not isinstance(raw_offer, Mapping):
        raise ProductOfferMappingError(
            "Raw product offer must be a dictionary-like mapping."
        )

    # External values are untyped at runtime. After confirming the container shape,
    # use a narrow mapping type for the explicit string-key lookups below.
    raw_mapping = cast(Mapping[str, object], raw_offer)
    mapped_offer = {
        target_field: raw_mapping[raw_field]
        for raw_field, target_field in RAW_FIELD_MAP.items()
        if raw_field in raw_mapping
    }

    if "unit" in mapped_offer:
        mapped_offer["unit"] = _normalize_unit(mapped_offer["unit"])

    promotion_status = _map_promotion_status(raw_mapping)
    if promotion_status is not None:
        mapped_offer["promotion_status"] = promotion_status

    try:
        return ProductOffer.model_validate(mapped_offer)
    except ValidationError as exc:
        # Summarize normalized field locations for callers while retaining Pydantic's
        # detailed errors as the exception cause for logs and debugging. Root-level
        # model validators have no field location, so include their message to make
        # cross-field failures such as invalid date ranges understandable to callers.
        invalid_fields = sorted(
            {
                str(error["loc"][0]) if error["loc"] else "offer"
                for error in exc.errors()
            }
        )
        model_errors = sorted(
            {str(error["msg"]) for error in exc.errors() if not error["loc"]}
        )
        error_summary = ", ".join(invalid_fields)
        if model_errors:
            error_summary += " (" + "; ".join(model_errors) + ")"
        raise ProductOfferMappingError(
            "Invalid raw product offer data for: " + error_summary + "."
        ) from exc
