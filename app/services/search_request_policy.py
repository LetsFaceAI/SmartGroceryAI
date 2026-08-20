"""Prepare deduplicated, cost-bounded search plans before external execution.

This application service is deliberately independent of MCP, LangChain tools, and
agents. Future orchestration should consume its immutable plan and call budget,
then invoke an application-owned search capability rather than a raw MCP tool.
"""

from collections.abc import Hashable, Iterator
from contextlib import contextmanager

from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.schemas.search_policy import SearchRequestPlan
from app.schemas.shopping import ShoppingItem, ShoppingRequest
from app.services.product_normalization import normalize_product_name


class SearchRequestPolicyError(ValueError):
    """Report a request that exceeds deterministic search safety policy."""


class ExternalActorBudgetExceededError(RuntimeError):
    """Report an attempted call or concurrency level outside the approved budget."""


def _normalize_optional_text(value: str | None) -> str:
    """Normalize optional text only for equivalence-key construction."""
    return "" if value is None else normalize_product_name(value)


def _item_equivalence_key(item: ShoppingItem) -> tuple[Hashable, ...]:
    """Identify requests that differ only by quantity, casing, or whitespace."""
    constraints = tuple(
        sorted(
            (
                normalize_product_name(constraint.value),
                constraint.requirement.value,
            )
            for constraint in item.constraints
        )
    )
    return (
        normalize_product_name(item.name),
        _normalize_optional_text(item.unit),
        _normalize_optional_text(item.notes),
        constraints,
    )


def _deduplicate_items(items: list[ShoppingItem]) -> tuple[ShoppingItem, ...]:
    """Merge equivalent item quantities while retaining the first validated text."""
    unique_items: dict[tuple[Hashable, ...], ShoppingItem] = {}
    for item in items:
        key = _item_equivalence_key(item)
        existing = unique_items.get(key)
        if existing is None:
            unique_items[key] = item
            continue

        try:
            # model_validate reruns quantity limits; model_copy(update=...) would
            # bypass validation and could create a plan with an invalid total.
            unique_items[key] = ShoppingItem.model_validate(
                {
                    **existing.model_dump(),
                    "quantity": existing.quantity + item.quantity,
                }
            )
        except ValidationError as exc:
            raise SearchRequestPolicyError(
                "Combined quantity for an equivalent requested item is invalid."
            ) from exc
    return tuple(unique_items.values())


def build_search_request_plan(
    request: ShoppingRequest,
    *,
    settings: Settings | None = None,
) -> SearchRequestPlan:
    """Deduplicate a shopping request and attach non-negotiable execution limits."""
    resolved_settings = settings or get_settings()
    unique_items = _deduplicate_items(request.items)
    if len(unique_items) > resolved_settings.search_max_items_per_request:
        raise SearchRequestPolicyError(
            "The request exceeds the configured maximum searchable item count."
        )

    return SearchRequestPlan(
        items=unique_items,
        request_preferences=tuple(request.preferences),
        max_external_actor_calls=(
            resolved_settings.search_max_external_actor_calls_per_request
        ),
        max_concurrency=resolved_settings.search_max_concurrency,
    )


class ExternalActorCallBudget:
    """Consume Actor-call and concurrency allowances without retry behavior."""

    def __init__(self, plan: SearchRequestPlan) -> None:
        self._max_calls = plan.max_external_actor_calls
        self._max_concurrency = plan.max_concurrency
        self._calls_used = 0
        self._active_calls = 0

    @property
    def calls_used(self) -> int:
        """Return the number of paid-call slots already consumed."""
        return self._calls_used

    @property
    def remaining_calls(self) -> int:
        """Return the remaining paid-call allowance."""
        return self._max_calls - self._calls_used

    @contextmanager
    def actor_call(self) -> Iterator[int]:
        """Reserve one bounded call slot for a future application-owned executor.

        Entering consumes the call permanently, even when execution fails. This is
        intentional because a timed-out Actor may still be running and incurring
        cost; the budget must never assume that failure makes a retry free.
        """
        if self._calls_used >= self._max_calls:
            raise ExternalActorBudgetExceededError(
                "The external Actor call budget for this request is exhausted."
            )
        if self._active_calls >= self._max_concurrency:
            raise ExternalActorBudgetExceededError(
                "The external Actor concurrency limit for this request is reached."
            )

        self._calls_used += 1
        self._active_calls += 1
        call_number = self._calls_used
        try:
            yield call_number
        finally:
            self._active_calls -= 1
