"""Immutable contracts for bounded grocery flyer search execution."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.shopping import Preference, ShoppingItem


class SearchRequestPlan(BaseModel):
    """Capture the request data and hard limits future executors must obey.

    This contract contains domain data and numeric budgets only. In particular, it
    never contains raw MCP tools that an agent could invoke outside application
    policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[ShoppingItem, ...] = Field(min_length=1, max_length=100)
    request_preferences: tuple[Preference, ...] = Field(default=(), max_length=20)
    max_external_actor_calls: int = Field(ge=1, le=10)
    max_concurrency: int = Field(ge=1, le=5)
    # Paid retries are intentionally not configurable. A retry must be a new,
    # explicitly budgeted call initiated by application code or the user.
    automatic_paid_retries: Literal[0] = 0
