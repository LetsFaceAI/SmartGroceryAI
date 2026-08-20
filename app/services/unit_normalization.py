"""Shared deterministic aliases for external and canonical unit boundaries.

Both raw-offer mapping and package parsing receive human-readable unit spellings.
Keeping their vocabulary here prevents the two layers from accepting different
aliases for the same ``MeasurementUnit`` contract.
"""

import unicodedata

from app.schemas.product_offer import MeasurementUnit

UNIT_ALIASES: dict[str, MeasurementUnit] = {
    "ea": MeasurementUnit.EACH,
    "each": MeasurementUnit.EACH,
    "ct": MeasurementUnit.COUNT,
    "count": MeasurementUnit.COUNT,
    "counts": MeasurementUnit.COUNT,
    "pk": MeasurementUnit.PACK,
    "pack": MeasurementUnit.PACK,
    "packs": MeasurementUnit.PACK,
    "g": MeasurementUnit.GRAM,
    "gram": MeasurementUnit.GRAM,
    "grams": MeasurementUnit.GRAM,
    "kg": MeasurementUnit.KILOGRAM,
    "kilogram": MeasurementUnit.KILOGRAM,
    "kilograms": MeasurementUnit.KILOGRAM,
    "ml": MeasurementUnit.MILLILITRE,
    "millilitre": MeasurementUnit.MILLILITRE,
    "millilitres": MeasurementUnit.MILLILITRE,
    "milliliter": MeasurementUnit.MILLILITRE,
    "milliliters": MeasurementUnit.MILLILITRE,
    "l": MeasurementUnit.LITRE,
    "litre": MeasurementUnit.LITRE,
    "litres": MeasurementUnit.LITRE,
    "liter": MeasurementUnit.LITRE,
    "liters": MeasurementUnit.LITRE,
    "oz": MeasurementUnit.OUNCE,
    "ounce": MeasurementUnit.OUNCE,
    "ounces": MeasurementUnit.OUNCE,
    "lb": MeasurementUnit.POUND,
    "lbs": MeasurementUnit.POUND,
    "pound": MeasurementUnit.POUND,
    "pounds": MeasurementUnit.POUND,
}


class UnsupportedMeasurementUnitError(ValueError):
    """Report a unit spelling that is outside the explicit alias vocabulary."""


def resolve_measurement_unit(unit: MeasurementUnit | str) -> MeasurementUnit:
    """Resolve a validated enum or explicit alias without semantic guessing."""
    if isinstance(unit, MeasurementUnit):
        return unit
    normalized = unicodedata.normalize("NFKC", unit).strip().casefold()
    try:
        return UNIT_ALIASES[normalized]
    except KeyError:
        raise UnsupportedMeasurementUnitError(
            f"Unsupported measurement unit: {unit!r}."
        ) from None
