"""Deterministically normalize validated offers for later product comparison.

This boundary intentionally performs no fuzzy matching or brand inference. Its
small alias tables, package grammar, and metric scale conversions are explicit so
future AI-assisted matching receives predictable inputs.
"""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from app.schemas.normalized_product import CanonicalUnit, NormalizedProduct
from app.schemas.product_offer import MeasurementUnit, ProductOffer
from app.services.unit_normalization import (
    UnsupportedMeasurementUnitError,
    resolve_measurement_unit,
)

# Symbols commonly appended by packaging and advertising systems carry no product
# identity. Other punctuation is retained because values such as "2%" and
# hyphenated varieties can be meaningful.
_FORMATTING_NOISE = str.maketrans("", "", "®™©\u200b\ufeff")
_EDGE_NOISE = " -_|•·"

# Aliases are intentionally small and reviewable. Unknown stores still receive
# safe text normalization, but are never guessed to be a known chain.
STORE_ALIASES: dict[str, str] = {
    "wal-mart": "walmart",
    "walmart canada": "walmart",
    "no frills": "no frills",
    "nofrills": "no frills",
    "real canadian superstore": "real canadian superstore",
    "rcss": "real canadian superstore",
}

_PACKAGE_PATTERN = re.compile(
    r"^(?:(?P<quantity>\d+)\s*[x×]\s*)?"
    r"(?P<size>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>[A-Za-z]+)$"
)


class ProductNormalizationError(ValueError):
    """Report input that cannot be normalized without making an unsafe guess."""


@dataclass(frozen=True, slots=True)
class NormalizedPackageSize:
    """Represent a parsed package using exact canonical quantities."""

    package_quantity: int
    package_size: Decimal
    total_package_size: Decimal
    unit: CanonicalUnit


def normalize_product_name(product_name: str) -> str:
    """Create a stable name using Unicode, whitespace, and case normalization.

    Punctuation and word order remain unchanged because altering them would begin
    product matching rather than deterministic normalization.
    """
    # Remove symbols before NFKC because compatibility normalization expands
    # characters such as ™ into letters ("TM"), which would create false tokens.
    without_noise = product_name.translate(_FORMATTING_NOISE)
    unicode_normalized = unicodedata.normalize("NFKC", without_noise)
    without_noise = unicode_normalized.translate(_FORMATTING_NOISE)
    return " ".join(without_noise.casefold().split()).strip(_EDGE_NOISE)


def normalize_store_name(store_name: str) -> str:
    """Normalize store text and apply only explicit, well-known aliases."""
    normalized = normalize_product_name(store_name)
    return STORE_ALIASES.get(normalized, normalized)


def normalize_unit(unit: MeasurementUnit | str) -> MeasurementUnit:
    """Resolve a unit enum or supported spelling without inferring unknown units."""
    try:
        return resolve_measurement_unit(unit)
    except UnsupportedMeasurementUnitError as exc:
        raise ProductNormalizationError(str(exc)) from None


def _canonicalize_package(
    package_quantity: int,
    package_size: Decimal,
    unit: MeasurementUnit,
) -> NormalizedPackageSize:
    """Apply only simple metric scale conversions to one validated package."""
    if package_quantity < 1 or package_size <= 0:
        raise ProductNormalizationError(
            "Package quantity and size must be greater than zero."
        )

    if unit is MeasurementUnit.KILOGRAM:
        canonical_size = package_size * Decimal("1000")
        canonical_unit = CanonicalUnit.GRAM
    elif unit is MeasurementUnit.LITRE:
        canonical_size = package_size * Decimal("1000")
        canonical_unit = CanonicalUnit.MILLILITRE
    else:
        canonical_size = package_size
        canonical_unit = CanonicalUnit(unit.value)

    return NormalizedPackageSize(
        package_quantity=package_quantity,
        package_size=canonical_size,
        total_package_size=canonical_size * Decimal(package_quantity),
        unit=canonical_unit,
    )


def parse_package_size(value: str) -> NormalizedPackageSize:
    """Parse a deliberately small package grammar into canonical exact values.

    Accepted forms are ``<size> <unit>`` and
    ``<quantity> x <size> <unit>``. Anything else fails clearly rather than
    guessing whether a number describes weight, volume, or item count.
    """
    normalized = unicodedata.normalize("NFKC", value).strip()
    match = _PACKAGE_PATTERN.fullmatch(normalized)
    if match is None:
        raise ProductNormalizationError(
            "Unsupported package size format. Expected values such as '500 g' "
            "or '2 x 500 g'."
        )

    package_quantity = int(match.group("quantity") or "1")
    package_size = Decimal(match.group("size"))
    unit = normalize_unit(match.group("unit"))
    return _canonicalize_package(package_quantity, package_size, unit)


def normalize_product_offer(offer: ProductOffer) -> NormalizedProduct:
    """Project one validated external offer into the canonical product model."""
    normalized_name = normalize_product_name(offer.product_name)
    normalized_store = normalize_store_name(offer.store)
    if not any(character.isalnum() for character in normalized_name):
        raise ProductNormalizationError(
            "The product name contains no comparable letters or numbers."
        )
    if not any(character.isalnum() for character in normalized_store):
        raise ProductNormalizationError(
            "The store name contains no comparable letters or numbers."
        )

    normalized_brand = None
    if offer.brand is not None:
        brand_candidate = normalize_product_name(offer.brand)
        # A formatting-only optional brand contains no reliable information, so
        # preserve absence rather than emitting an empty comparison value.
        if any(character.isalnum() for character in brand_candidate):
            normalized_brand = brand_candidate

    package = None
    if offer.package_size is not None and offer.unit is not None:
        package = _canonicalize_package(1, offer.package_size, offer.unit)

    return NormalizedProduct(
        normalized_name=normalized_name,
        brand=normalized_brand,
        store=normalized_store,
        package_quantity=package.package_quantity if package else None,
        package_size=package.package_size if package else None,
        total_package_size=package.total_package_size if package else None,
        unit=package.unit if package else None,
        price=offer.price,
        regular_price=offer.regular_price,
        currency=offer.currency,
        promotion_status=offer.promotion_status,
        original_offer=offer,
    )
