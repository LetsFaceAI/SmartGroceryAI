"""Deterministically normalize validated offers for later product comparison.

This first normalization boundary intentionally performs no fuzzy matching, brand
inference, package parsing, or unit conversion. Keeping those policies out makes
the current output predictable and gives future AI-assisted matching a safe input.
"""

import unicodedata

from app.schemas.normalized_product import NormalizedProduct
from app.schemas.product_offer import ProductOffer


def normalize_product_name(product_name: str) -> str:
    """Create a stable name using Unicode, whitespace, and case normalization.

    Punctuation and word order remain unchanged because altering them would begin
    product matching rather than deterministic normalization.
    """
    unicode_normalized = unicodedata.normalize("NFKC", product_name)
    return " ".join(unicode_normalized.casefold().split())


def normalize_product_offer(offer: ProductOffer) -> NormalizedProduct:
    """Project one validated external offer into the canonical product model."""
    return NormalizedProduct(
        normalized_name=normalize_product_name(offer.product_name),
        brand=offer.brand,
        store=offer.store,
        package_size=offer.package_size,
        unit=offer.unit,
        price=offer.price,
        regular_price=offer.regular_price,
        currency=offer.currency,
        promotion_status=offer.promotion_status,
        original_offer=offer,
    )
