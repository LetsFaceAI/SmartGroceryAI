"""Parse the small package-size grammar shared by ingestion and normalization.

The parser preserves the advertised per-package size and multiplier. Canonical
metric scaling remains the responsibility of product normalization so this module
does not blur external-data validation with comparison representation.
"""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from app.schemas.product_offer import MeasurementUnit
from app.services.unit_normalization import (
    UnsupportedMeasurementUnitError,
    resolve_measurement_unit,
)

_PACKAGE_PATTERN = re.compile(
    r"^(?:(?P<quantity>\d+)\s*[x×]\s*)?"
    r"(?P<size>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>[A-Za-z]+)$"
)


class PackageSizeParsingError(ValueError):
    """Report a package string that cannot be interpreted without guessing."""


@dataclass(frozen=True, slots=True)
class ParsedPackageSize:
    """Retain one parsed multiplier, per-package size, and validated unit."""

    package_quantity: int
    package_size: Decimal
    unit: MeasurementUnit


def parse_external_package_size(value: str) -> ParsedPackageSize:
    """Parse ``<size> <unit>`` or ``<quantity> x <size> <unit>`` exactly."""
    normalized = unicodedata.normalize("NFKC", value).strip()
    match = _PACKAGE_PATTERN.fullmatch(normalized)
    if match is None:
        raise PackageSizeParsingError(
            "Unsupported package size format. Expected values such as '500 g' "
            "or '2 x 500 g'."
        )

    package_quantity = int(match.group("quantity") or "1")
    package_size = Decimal(match.group("size"))
    if package_quantity < 1 or package_size <= 0:
        raise PackageSizeParsingError(
            "Package quantity and size must be greater than zero."
        )

    try:
        unit = resolve_measurement_unit(match.group("unit"))
    except UnsupportedMeasurementUnitError as exc:
        raise PackageSizeParsingError(str(exc)) from None

    return ParsedPackageSize(
        package_quantity=package_quantity,
        package_size=package_size,
        unit=unit,
    )
