"""Conservative deterministic matching for requested and advertised products.

This service evaluates product identity and explicit user constraints only. Prices,
stores, and offer validity belong to the comparison layer after a product has
passed the trusted matching policy.
"""

import re
from collections.abc import Sequence

from app.schemas.normalized_product import NormalizedProduct
from app.schemas.product_match import (
    ProductMatchDecision,
    ProductMatchRequest,
    ProductMatchResult,
    ProductMatchType,
)
from app.schemas.shopping import (
    ConstraintRequirement,
    ShoppingConstraint,
    ShoppingItem,
)
from app.services.product_normalization import normalize_product_name

_TOKEN_PATTERN = re.compile(r"[^\W_]+%?", re.UNICODE)
_MIN_SINGLE_TOKEN_LENGTH = 4
_MIN_OVERLAP_TOKENS = 2
_MIN_TOKEN_JACCARD = 0.80

# These terms change the requested base product into a distinct grocery category.
# Containment therefore remains visible as a candidate, but is never auto-ranked.
_CATEGORY_CHANGING_TOKENS = {
    "beverage",
    "candy",
    "chocolate",
    "drink",
    "juice",
    "sauce",
    "syrup",
}

# Only this small vocabulary is inferred from legacy notes and global preferences.
# Arbitrary prose is preserved on ProductMatchRequest, not guessed.
_PROTECTED_CONSTRAINT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("organic", re.compile(r"\borganic\b", re.IGNORECASE)),
    ("2%", re.compile(r"(?<!\w)2\s*%(?!\w)", re.IGNORECASE)),
    ("lactose-free", re.compile(r"\blactose[\s-]*free\b", re.IGNORECASE)),
    ("gluten-free", re.compile(r"\bgluten[\s-]*free\b", re.IGNORECASE)),
)
_OPTIONAL_LANGUAGE_PATTERN = re.compile(
    r"\b(?:prefer(?:red)?|if\s+(?:available|possible)|"
    r"when\s+(?:available|possible))\b",
    re.IGNORECASE,
)
_NOTE_CLAUSE_SPLIT_PATTERN = re.compile(
    r"\s*(?:[,;]|\b(?:and|but)\b)\s*",
    re.IGNORECASE,
)


class ProductMatchingError(ValueError):
    """Report a requested name that cannot be matched safely."""


def _tokenize(value: str) -> tuple[str, ...]:
    """Return comparable word tokens while retaining numeric percentages."""
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


def _is_containment_candidate(
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


def _extract_protected_constraints(value: str | None) -> tuple[str, ...]:
    """Extract only explicitly supported qualifiers from unstructured text."""
    if value is None:
        return ()
    return tuple(
        canonical
        for canonical, pattern in _PROTECTED_CONSTRAINT_PATTERNS
        if pattern.search(value)
    )


def _canonical_constraint(value: str) -> str:
    """Normalize known spelling variants while preserving other explicit text."""
    recognized = _extract_protected_constraints(value)
    if len(recognized) == 1:
        return recognized[0]
    return normalize_product_name(value)


def build_product_match_request(
    item: ShoppingItem,
    *,
    request_preferences: Sequence[str] = (),
) -> ProductMatchRequest:
    """Preserve a ShoppingItem and derive only known legacy-note constraints.

    Explicit item constraints always win. Known qualifiers in the item name default
    to required. Legacy notes default to required unless they contain a small,
    explicit optional phrase; request-wide qualifiers default to optional.
    """
    constraints: dict[str, ShoppingConstraint] = {}
    for constraint in item.constraints:
        canonical = _canonical_constraint(constraint.value)
        constraints[canonical] = ShoppingConstraint(
            value=canonical,
            requirement=constraint.requirement,
        )

    for value in _extract_protected_constraints(item.name):
        constraints.setdefault(
            value,
            ShoppingConstraint(
                value=value,
                requirement=ConstraintRequirement.REQUIRED,
            ),
        )

    if item.notes is not None:
        # Clause-local optional markers avoid weakening a separate requirement in
        # notes such as "must be lactose-free, but prefer organic if available."
        for clause in _NOTE_CLAUSE_SPLIT_PATTERN.split(item.notes):
            note_requirement = (
                ConstraintRequirement.OPTIONAL
                if _OPTIONAL_LANGUAGE_PATTERN.search(clause)
                else ConstraintRequirement.REQUIRED
            )
            for value in _extract_protected_constraints(clause):
                constraints.setdefault(
                    value,
                    ShoppingConstraint(value=value, requirement=note_requirement),
                )

    for preference in request_preferences:
        for value in _extract_protected_constraints(preference):
            constraints.setdefault(
                value,
                ShoppingConstraint(
                    value=value,
                    requirement=ConstraintRequirement.OPTIONAL,
                ),
            )

    return ProductMatchRequest(
        shopping_item=item,
        effective_constraints=tuple(constraints.values()),
        request_preferences=tuple(request_preferences),
    )


def _constraint_outcomes(
    request: ProductMatchRequest,
    product: NormalizedProduct,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return missing required and optional constraints without inferring claims."""
    # A brand name containing "organic" is not proof that this specific product is
    # organic, so only the advertised product name can satisfy a constraint.
    searchable_text = product.normalized_name
    product_constraints = set(_extract_protected_constraints(searchable_text))
    product_tokens = _tokenize(searchable_text)
    missing_required: list[str] = []
    missing_optional: list[str] = []

    for constraint in request.constraints:
        canonical = _canonical_constraint(constraint.value)
        satisfied = canonical in product_constraints or _contains_token_sequence(
            product_tokens,
            _tokenize(canonical),
        )
        if satisfied:
            continue
        target = (
            missing_required
            if constraint.requirement is ConstraintRequirement.REQUIRED
            else missing_optional
        )
        target.append(canonical)

    return tuple(missing_required), tuple(missing_optional)


def _result(
    *,
    request: ProductMatchRequest,
    product: NormalizedProduct,
    decision: ProductMatchDecision,
    match_type: ProductMatchType,
    confidence: float,
    reason: str,
    missing_required: tuple[str, ...] = (),
    missing_optional: tuple[str, ...] = (),
) -> ProductMatchResult:
    """Build one internally consistent result from a policy decision."""
    is_safe = decision is ProductMatchDecision.SAFE
    is_candidate = decision is not ProductMatchDecision.NO_MATCH
    return ProductMatchResult(
        matched=is_safe,
        candidate=is_candidate,
        safe_for_ranking=is_safe,
        decision=decision,
        match_type=match_type,
        confidence=confidence,
        reason=reason,
        normalized_request_name=normalize_product_name(request.item_name),
        request=request,
        unmet_required_constraints=missing_required,
        unmet_optional_constraints=missing_optional,
        product=product,
    )


def match_product(
    requested_item: str | ProductMatchRequest | ShoppingItem,
    product: NormalizedProduct,
) -> ProductMatchResult:
    """Compare a requested item with one normalized flyer product.

    Exact and safe containment matches may be ranked automatically. Category-
    changing containment, strong token overlap, or missing required qualifiers are
    retained as explicit uncertain candidates for later user/AI review.
    """
    if isinstance(requested_item, ShoppingItem):
        request = build_product_match_request(requested_item)
    elif isinstance(requested_item, ProductMatchRequest):
        request = requested_item
    else:
        if not any(character.isalnum() for character in requested_item):
            raise ProductMatchingError(
                "The requested item name must contain at least one letter or number."
            )
        request = ProductMatchRequest(shopping_item=ShoppingItem(name=requested_item))

    normalized_request = normalize_product_name(request.item_name)
    requested_tokens = _tokenize(normalized_request)
    product_tokens = _tokenize(product.normalized_name)
    if not requested_tokens:
        raise ProductMatchingError(
            "The requested item name must contain at least one letter or number."
        )

    missing_required, missing_optional = _constraint_outcomes(request, product)
    if normalized_request == product.normalized_name:
        match_type = ProductMatchType.EXACT
        confidence = 1.0
        reason = "The normalized product names are identical."
    elif _is_containment_candidate(requested_tokens, product_tokens):
        match_type = ProductMatchType.CONTAINMENT
        confidence = 0.85
        extra_tokens = set(product_tokens) - set(requested_tokens)
        if extra_tokens & _CATEGORY_CHANGING_TOKENS:
            return _result(
                request=request,
                product=product,
                decision=ProductMatchDecision.UNCERTAIN,
                match_type=match_type,
                confidence=confidence,
                reason=(
                    "The product contains the requested name plus a term that may "
                    "change its grocery category."
                ),
                missing_required=missing_required,
                missing_optional=missing_optional,
            )
        reason = "The product name contains the complete requested token sequence."
    else:
        requested_set = set(requested_tokens)
        product_set = set(product_tokens)
        shared_tokens = requested_set & product_set
        union_tokens = requested_set | product_set
        jaccard_score = len(shared_tokens) / len(union_tokens) if union_tokens else 0.0
        if (
            len(shared_tokens) >= _MIN_OVERLAP_TOKENS
            and jaccard_score >= _MIN_TOKEN_JACCARD
        ):
            return _result(
                request=request,
                product=product,
                decision=ProductMatchDecision.UNCERTAIN,
                match_type=ProductMatchType.TOKEN_OVERLAP,
                confidence=0.70,
                reason=(
                    "The names have strong token overlap, but reordered words are "
                    "not trusted as automatic equivalence."
                ),
                missing_required=missing_required,
                missing_optional=missing_optional,
            )
        return _result(
            request=request,
            product=product,
            decision=ProductMatchDecision.NO_MATCH,
            match_type=ProductMatchType.NONE,
            confidence=0.0,
            reason="The names did not satisfy the conservative deterministic rules.",
            missing_required=missing_required,
            missing_optional=missing_optional,
        )

    if missing_required:
        return _result(
            request=request,
            product=product,
            decision=ProductMatchDecision.UNCERTAIN,
            match_type=match_type,
            confidence=confidence,
            reason="The name matched, but required product qualifiers were not proven.",
            missing_required=missing_required,
            missing_optional=missing_optional,
        )

    return _result(
        request=request,
        product=product,
        decision=ProductMatchDecision.SAFE,
        match_type=match_type,
        confidence=confidence,
        reason=reason,
        missing_optional=missing_optional,
    )
