"""Shared validation for geographic search inputs.

Provider adapters should receive the same canonical postal code so routing between
Apify and future web-search providers cannot change the meaning of a request.
"""

import re
from typing import Annotated

from pydantic import BeforeValidator, Field


def normalize_postal_code(value: object) -> object:
    """Validate US/Canadian postal codes and canonicalize Canadian spacing.

    The function accepts ``object`` because Pydantic runs it before string parsing.
    Non-string values are returned unchanged so Pydantic can report the normal type
    error instead of this validator hiding it behind a misleading format error.
    """
    if not isinstance(value, str):
        return value

    stripped_value = value.strip()
    compact_value = re.sub(r"\s+", "", stripped_value)
    if re.fullmatch(r"[A-Za-z]\d[A-Za-z]\d[A-Za-z]\d", compact_value):
        uppercase_value = compact_value.upper()
        return f"{uppercase_value[:3]} {uppercase_value[3:]}"

    if not re.fullmatch(r"\d{5}(?:-\d{4})?", stripped_value):
        raise ValueError("postal_code must be a Canadian postal code or US ZIP code.")
    return stripped_value


# Reusing this alias keeps provider-neutral requests and provider-specific payloads
# from drifting into different validation or normalization behavior.
PostalCode = Annotated[
    str,
    BeforeValidator(normalize_postal_code),
    Field(min_length=3, max_length=10),
]
