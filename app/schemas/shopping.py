"""Pydantic schemas for normalized grocery shopping requests.

These models establish the data shape that later structured-output flows will ask
an LLM to produce. They contain only domain data and validation; no model invocation
or LangChain behavior belongs in this module.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Preferences are short user constraints such as "organic" or "no substitutions".
# Validating each list element prevents empty strings from becoming meaningless data.
Preference = Annotated[str, Field(min_length=1, max_length=100)]


class ShoppingItem(BaseModel):
    """Represent one normalized item in a user's grocery request.

    Quantity uses a float because groceries may be requested in fractional amounts,
    such as 1.5 kilograms. Unit remains optional because requests like "bananas" may
    not include one and should still be valid.
    """

    # Trimming strings handles natural-language whitespace. Forbidding unknown fields
    # catches misspelled keys instead of silently discarding user or LLM output.
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=120,
        description="Common grocery item name, such as milk or brown rice.",
    )
    quantity: float = Field(
        default=1.0,
        gt=0,
        le=10_000,
        description="Positive amount requested by the user.",
    )
    unit: str | None = Field(
        default=None,
        min_length=1,
        max_length=30,
        description="Optional unit such as kg, cartons, or packages.",
    )
    notes: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
        description="Optional item-specific details or acceptable substitutions.",
    )


class ShoppingRequest(BaseModel):
    """Represent a complete, validated grocery shopping request."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    items: list[ShoppingItem] = Field(
        min_length=1,
        max_length=100,
        description="One or more normalized grocery items to shop for.",
    )
    preferences: list[Preference] = Field(
        default_factory=list,
        max_length=20,
        description="Request-wide preferences such as organic or lowest price.",
    )
    notes: str | None = Field(
        default=None,
        min_length=1,
        max_length=1_000,
        description="Optional context that applies to the entire shopping request.",
    )
