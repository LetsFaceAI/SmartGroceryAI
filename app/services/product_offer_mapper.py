"""Map simple external flyer records into validated ``ProductOffer`` objects.

The raw keys below are intentionally provider-neutral placeholders. When the Apify
MCP response shape is known, this module can change its key and value normalization
without forcing comparison, planning, or AI code to understand that external shape.
"""

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import cast

from pydantic import ValidationError

from app.schemas.product_offer import ProductOffer, PromotionStatus
from app.services.package_size_parser import (
    PackageSizeParsingError,
    ParsedPackageSize,
    parse_external_package_size,
)
from app.services.unit_normalization import (
    UnsupportedMeasurementUnitError,
    resolve_measurement_unit,
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
    "packageQuantity": "package_quantity",
    "unit": "unit",
    "validFrom": "valid_from",
    "validUntil": "valid_until",
    "source": "source",
}


class ProductOfferMappingError(ValueError):
    """Report raw flyer data that cannot become a valid product offer."""


def _normalize_unit(value: object) -> object:
    """Translate a known unit alias while leaving invalid values for validation."""
    if isinstance(value, str):
        try:
            return resolve_measurement_unit(value)
        except UnsupportedMeasurementUnitError:
            # The mapper preserves its existing behavior: Pydantic owns the final
            # external-contract error and reports the invalid ``unit`` field.
            return value.strip()
    return value


def _parse_embedded_package_size(
    raw_mapping: Mapping[str, object],
    mapped_offer: dict[str, object],
) -> None:
    """Retain multipack details when packageSize includes its unit and quantity."""
    raw_package_size = raw_mapping.get("packageSize")
    if not isinstance(raw_package_size, str):
        return

    try:
        parsed = parse_external_package_size(raw_package_size)
    except PackageSizeParsingError:
        # Numeric strings with a separate unit remain valid existing input. Other
        # malformed strings are left for ProductOffer to reject as package_size.
        return

    _validate_embedded_package_consistency(raw_mapping, parsed)
    mapped_offer["package_quantity"] = parsed.package_quantity
    mapped_offer["package_size"] = parsed.package_size
    mapped_offer["unit"] = parsed.unit


def _validate_embedded_package_consistency(
    raw_mapping: Mapping[str, object],
    parsed: ParsedPackageSize,
) -> None:
    """Reject conflicting duplicate package metadata instead of choosing one."""
    raw_quantity = raw_mapping.get("packageQuantity")
    if raw_quantity is not None:
        try:
            numeric_quantity = Decimal(str(raw_quantity))
        except InvalidOperation:
            numeric_quantity = Decimal("-1")
        if (
            not numeric_quantity.is_finite()
            or numeric_quantity != numeric_quantity.to_integral_value()
            or int(numeric_quantity) != parsed.package_quantity
        ):
            raise ProductOfferMappingError(
                "Raw packageQuantity conflicts with the quantity in packageSize."
            )

    raw_unit = raw_mapping.get("unit")
    if raw_unit is None:
        return
    try:
        resolved_unit = (
            resolve_measurement_unit(raw_unit)
            if isinstance(raw_unit, str)
            else raw_unit
        )
    except UnsupportedMeasurementUnitError as exc:
        raise ProductOfferMappingError("Raw unit conflicts with packageSize.") from exc
    if resolved_unit != parsed.unit:
        raise ProductOfferMappingError("Raw unit conflicts with packageSize.")


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

    _parse_embedded_package_size(raw_mapping, mapped_offer)

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
