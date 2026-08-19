"""Transform Flipp Actor records into the application's product-offer contract.

This provider-specific layer understands Apify/Flipp field names. It deliberately
delegates generic field mapping and all business validation to
``map_product_offer`` so external response changes remain isolated here.
"""

import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import cast

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.schemas.flyer_search import RawFlyerSearchRequest, RawFlyerSearchResult
from app.schemas.product_offer import ProductOffer
from app.services.product_offer_mapper import (
    ProductOfferMappingError,
    map_product_offer,
)
from app.services.raw_flyer_search import search_raw_flyer_offers

# The Actor's search output uses these keys for the values required by
# ProductOffer. Keeping this map explicit makes future Actor schema changes easy
# to review and prevents arbitrary provider metadata from leaking into the app.
FLIPP_FIELD_MAP: dict[str, str] = {
    "name": "product",
    "brand": "brand",
    "merchantName": "store",
    "currentPrice": "price",
    "originalPrice": "regularPrice",
    "validFrom": "validFrom",
    "validTo": "validUntil",
    "sourceUrl": "source",
}

_CANADIAN_POSTAL_CODE = re.compile(r"^[A-Z]\d[A-Z][ -]?\d[A-Z]\d$", re.IGNORECASE)
_US_ZIP_CODE = re.compile(r"^\d{5}(?:-\d{4})?$")
_RECORD_CONTAINER_KEYS = ("items", "results", "datasetItems", "dataset")


class ApifyFlyerTransformationError(ValueError):
    """Report an MCP response or Flipp record that cannot be safely normalized."""


def _currency_for_record(record: Mapping[str, object], postal_code: str) -> str:
    """Use an explicit record currency, otherwise infer it from validated geography."""
    record_currency = record.get("currency")
    if isinstance(record_currency, str) and record_currency.strip():
        return record_currency.strip().upper()
    if _CANADIAN_POSTAL_CODE.fullmatch(postal_code.strip()):
        return "CAD"
    if _US_ZIP_CODE.fullmatch(postal_code.strip()):
        return "USD"
    raise ApifyFlyerTransformationError(
        "Cannot determine offer currency from the record or postal code."
    )


def _sale_flag(record: Mapping[str, object]) -> bool | None:
    """Derive sale status only when both advertised prices make it unambiguous."""
    price = record.get("currentPrice")
    regular_price = record.get("originalPrice")
    if regular_price is None:
        return None

    # Decimal parsing and contradictory relationships remain the ProductOffer
    # contract's responsibility. This comparison is only a status hint.
    try:
        current = Decimal(str(price))
        regular = Decimal(str(regular_price))
    except (InvalidOperation, ValueError):
        return None
    if current < regular:
        return True
    if current == regular:
        return False
    return None


def map_apify_flyer_record(
    raw_record: object,
    *,
    postal_code: str,
) -> ProductOffer:
    """Convert one Actor dataset item through the existing ProductOffer mapper.

    Args:
        raw_record: One dictionary-like item from the Flipp Actor dataset.
        postal_code: Search location used to choose CAD or USD when the record
            does not expose a currency field.

    Raises:
        ApifyFlyerTransformationError: If the record shape, required data, or
            resulting ProductOffer is invalid.
    """
    if not isinstance(raw_record, Mapping):
        raise ApifyFlyerTransformationError(
            "A Flipp flyer result must be a dictionary-like record."
        )

    record = cast(Mapping[str, object], raw_record)
    mapped = {
        target: record[source]
        for source, target in FLIPP_FIELD_MAP.items()
        if source in record and record[source] is not None
    }
    mapped["currency"] = _currency_for_record(record, postal_code)

    sale_flag = _sale_flag(record)
    if sale_flag is not None:
        mapped["onSale"] = sale_flag

    # A URL is preferred, but an Actor item ID is still a stable, traceable source
    # when a partial dataset row omits sourceUrl.
    if "source" not in mapped:
        item_id = record.get("itemId")
        if isinstance(item_id, str) and item_id.strip():
            mapped["source"] = f"apify:flipp:{item_id.strip()}"

    try:
        return map_product_offer(mapped)
    except ProductOfferMappingError as exc:
        raise ApifyFlyerTransformationError(
            "Invalid Flipp flyer record: " + str(exc)
        ) from exc


def _parse_json_text(value: str) -> object | None:
    """Parse content blocks only when their entire text is valid JSON."""
    try:
        parsed: object = json.loads(value)
        return parsed
    except json.JSONDecodeError:
        return None


def _find_records(value: object) -> list[object] | None:
    """Find dataset items in common MCP content and structured-artifact wrappers."""
    if isinstance(value, str):
        parsed = _parse_json_text(value)
        return None if parsed is None else _find_records(parsed)

    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        if any(
            key in mapping
            for key in ("itemId", "merchantName", "currentPrice", "sourceUrl")
        ):
            return [mapping]

        for key in _RECORD_CONTAINER_KEYS:
            if key in mapping:
                candidate = mapping[key]
                if isinstance(candidate, Sequence) and not isinstance(
                    candidate, (str, bytes, bytearray)
                ):
                    return list(candidate)
                nested = _find_records(candidate)
                if nested is not None:
                    return nested

        # LangChain stores MCP structuredContent under artifact, while some mocks
        # and serialized responses use the protocol's camelCase spelling.
        for key in ("artifact", "structured_content", "structuredContent", "content"):
            if key in mapping:
                nested = _find_records(mapping[key])
                if nested is not None:
                    return nested
        return None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        # A direct list of record mappings is already a valid dataset response.
        if not value or all(isinstance(item, Mapping) for item in value):
            direct_items = list(value)
            if not direct_items or any(
                {"name", "merchantName", "currentPrice"}.intersection(item)
                for item in direct_items
                if isinstance(item, Mapping)
            ):
                return direct_items
        for item in value:
            nested = _find_records(item)
            if nested is not None:
                return nested
        return None

    # ToolMessage is a Pydantic model; model_dump preserves both content and
    # artifact without tying this transformer to one LangChain response class.
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return _find_records(dumped)
    return None


def transform_apify_flyer_result(
    result: RawFlyerSearchResult,
) -> list[ProductOffer]:
    """Extract and validate all preview records from one raw MCP search result."""
    records = _find_records(result.raw_response)
    if records is None:
        raise ApifyFlyerTransformationError(
            "The MCP response did not contain structured Flipp dataset items."
        )

    offers: list[ProductOffer] = []
    for index, record in enumerate(records):
        try:
            offers.append(
                map_apify_flyer_record(
                    record,
                    postal_code=result.request.postal_code,
                )
            )
        except ApifyFlyerTransformationError as exc:
            raise ApifyFlyerTransformationError(
                f"Invalid Flipp flyer record at index {index}: {exc}"
            ) from exc
    return offers


async def search_product_offers(
    request: RawFlyerSearchRequest,
    *,
    client: MultiServerMCPClient | None = None,
    timeout_seconds: float | None = None,
) -> list[ProductOffer]:
    """Run the existing bounded raw search once, then validate its preview items."""
    raw_result = await search_raw_flyer_offers(
        request,
        client=client,
        timeout_seconds=timeout_seconds,
    )
    return transform_apify_flyer_result(raw_result)
