"""Conservative deterministic matching for requested and advertised products.

This service intentionally evaluates product identity only. Prices, stores, and
promotion details must be handled by later comparison services after a product
has passed these name-based rules.
"""

import re

from app.schemas.normalized_product import NormalizedProduct
from app.schemas.product_match import ProductMatchResult, ProductMatchType
from app.services.product_normalization import normalize_product_name

_TOKEN_PATTERN = re.compile(r"[^\W_]+%?", re.UNICODE)
_MIN_SINGLE_TOKEN_LENGTH = 4
_MIN_OVERLAP_TOKENS = 2
_MIN_TOKEN_JACCARD = 0.80


class ProductMatchingError(ValueError):
    """Report a requested name that cannot be matched safely."""


def _tokenize(value: str) -> tuple[str, ...]:
    """Return comparable word tokens while retaining numeric percentage details."""
    return tuple(_TOKEN_PATTERN.findall(value))


def _contains_token_sequence(
    longer_tokens: tuple[str, ...],
    shorter_tokens: tuple[str, ...],
) -> bool:
    """Check contiguous whole-token containment instead of unsafe substrings."""
    window_size = len(shorter_tokens)
    if window_size == 0 or window_size > len(longer_tokens):
        return False
    return any(
        longer_tokens[index : index + window_size] == shorter_tokens
        for index in range(len(longer_tokens) - window_size + 1)
    )


def _is_safe_containment(
    requested_tokens: tuple[str, ...],
    product_tokens: tuple[str, ...],
) -> bool:
    """Require the complete request in the product without dropping qualifiers."""
    if (
        len(requested_tokens) == 1
        and len(requested_tokens[0]) < _MIN_SINGLE_TOKEN_LENGTH
    ):
        return False
    return _contains_token_sequence(product_tokens, requested_tokens)


def match_product(
    requested_item_name: str,
    product: NormalizedProduct,
) -> ProductMatchResult:
    """Compare one requested item with one normalized flyer product.

    Rules run from strongest to weakest. The function stops at the first match so
    every result has a stable explanation and confidence value.

    Raises:
        ProductMatchingError: If normalization leaves the request without words.
    """
    normalized_request = normalize_product_name(requested_item_name)
    requested_tokens = _tokenize(normalized_request)
    product_tokens = _tokenize(product.normalized_name)
    if not requested_tokens:
        raise ProductMatchingError(
            "The requested item name must contain at least one letter or number."
        )

    if normalized_request == product.normalized_name:
        return ProductMatchResult(
            matched=True,
            match_type=ProductMatchType.EXACT,
            confidence=1.0,
            reason="The normalized product names are identical.",
            normalized_request_name=normalized_request,
            product=product,
        )

    if _is_safe_containment(requested_tokens, product_tokens):
        return ProductMatchResult(
            matched=True,
            match_type=ProductMatchType.CONTAINMENT,
            confidence=0.85,
            reason="The product name contains the complete requested token sequence.",
            normalized_request_name=normalized_request,
            product=product,
        )

    requested_set = set(requested_tokens)
    product_set = set(product_tokens)
    shared_tokens = requested_set & product_set
    union_tokens = requested_set | product_set
    jaccard_score = len(shared_tokens) / len(union_tokens) if union_tokens else 0.0
    if (
        len(shared_tokens) >= _MIN_OVERLAP_TOKENS
        and jaccard_score >= _MIN_TOKEN_JACCARD
    ):
        return ProductMatchResult(
            matched=True,
            match_type=ProductMatchType.TOKEN_OVERLAP,
            confidence=0.70,
            reason=("The names share multiple words with at least 80% token overlap."),
            normalized_request_name=normalized_request,
            product=product,
        )

    return ProductMatchResult(
        matched=False,
        match_type=ProductMatchType.NONE,
        confidence=0.0,
        reason="The names did not satisfy the conservative deterministic rules.",
        normalized_request_name=normalized_request,
        product=product,
    )
