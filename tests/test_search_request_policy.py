"""Tests for deterministic request deduplication and external-call budgets."""

import pytest

from app.core.config import Settings
from app.schemas.search_policy import SearchRequestPlan
from app.schemas.shopping import ShoppingConstraint, ShoppingItem, ShoppingRequest
from app.services.search_request_policy import (
    ExternalActorBudgetExceededError,
    ExternalActorCallBudget,
    SearchRequestPolicyError,
    build_search_request_plan,
)


def make_settings(**overrides: object) -> Settings:
    """Create isolated settings without reading the developer's environment."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        **overrides,  # type: ignore[arg-type]
    )


def test_plan_deduplicates_equivalent_items_and_combines_quantity() -> None:
    """Equivalent casing and constraint order should consume one search-item slot."""
    request = ShoppingRequest(
        items=[
            ShoppingItem(
                name="Milk",
                quantity=1,
                unit="carton",
                constraints=(ShoppingConstraint(value="2%"),),
            ),
            ShoppingItem(
                name="milk",
                quantity=2,
                unit="CARTON",
                constraints=(ShoppingConstraint(value="2%"),),
            ),
            ShoppingItem(name="Eggs"),
        ],
        preferences=["lowest price"],
    )

    plan = build_search_request_plan(request, settings=make_settings())

    assert type(plan) is SearchRequestPlan
    assert [item.name for item in plan.items] == ["Milk", "Eggs"]
    assert plan.items[0].quantity == 3
    assert plan.request_preferences == ("lowest price",)


def test_deduplication_does_not_merge_different_constraints() -> None:
    """A required qualifier changes product identity for search-policy purposes."""
    request = ShoppingRequest(
        items=[
            ShoppingItem(name="milk"),
            ShoppingItem(
                name="milk",
                constraints=(ShoppingConstraint(value="organic"),),
            ),
        ]
    )

    plan = build_search_request_plan(request, settings=make_settings())

    assert len(plan.items) == 2


def test_plan_rejects_more_unique_items_than_configured() -> None:
    """Oversized requests must fail before any external operation can start."""
    request = ShoppingRequest(
        items=[ShoppingItem(name="milk"), ShoppingItem(name="eggs")]
    )

    with pytest.raises(SearchRequestPolicyError, match="maximum searchable"):
        build_search_request_plan(
            request,
            settings=make_settings(search_max_items_per_request=1),
        )


def test_plan_exposes_bounded_calls_concurrency_and_zero_retries() -> None:
    """Future executors should receive explicit non-negotiable cost limits."""
    plan = build_search_request_plan(
        ShoppingRequest(items=[ShoppingItem(name="milk")]),
        settings=make_settings(
            search_max_external_actor_calls_per_request=2,
            search_max_concurrency=1,
        ),
    )

    assert plan.max_external_actor_calls == 2
    assert plan.max_concurrency == 1
    assert plan.automatic_paid_retries == 0


def test_actor_budget_enforces_call_and_concurrency_limits() -> None:
    """Every attempt consumes a slot and nested calls cannot exceed concurrency."""
    plan = build_search_request_plan(
        ShoppingRequest(items=[ShoppingItem(name="milk")]),
        settings=make_settings(
            search_max_external_actor_calls_per_request=2,
            search_max_concurrency=1,
        ),
    )
    budget = ExternalActorCallBudget(plan)

    with budget.actor_call() as first_call:
        assert first_call == 1
        with pytest.raises(ExternalActorBudgetExceededError, match="concurrency"):
            with budget.actor_call():
                pass

    with budget.actor_call() as second_call:
        assert second_call == 2

    assert budget.calls_used == 2
    assert budget.remaining_calls == 0
    with pytest.raises(ExternalActorBudgetExceededError, match="exhausted"):
        with budget.actor_call():
            pass


def test_failed_execution_still_consumes_actor_call_slot() -> None:
    """A local failure cannot make a possibly running paid Actor call free."""
    plan = build_search_request_plan(
        ShoppingRequest(items=[ShoppingItem(name="milk")]),
        settings=make_settings(search_max_external_actor_calls_per_request=1),
    )
    budget = ExternalActorCallBudget(plan)

    with pytest.raises(RuntimeError, match="simulated failure"):
        with budget.actor_call():
            raise RuntimeError("simulated failure")

    assert budget.remaining_calls == 0
