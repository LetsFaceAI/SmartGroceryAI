"""Transform Flipp Actor records into the application's product-offer contract.

This provider-specific layer understands Apify/Flipp field names. It deliberately
delegates generic field mapping and all business validation to
``map_product_offer`` so external response changes remain isolated here.
"""

import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import cast

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import get_settings
from app.core.mcp import create_apify_mcp_client
from app.schemas.flyer_search import RawFlyerSearchRequest, RawFlyerSearchResult
from app.schemas.product_offer import ProductOffer
from app.services.apify_dataset_reader import (
    ApifyDatasetReadError,
    extract_default_dataset_id,
    fetch_apify_dataset_items,
)
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


class MissingFlyerDatasetItemsError(ApifyFlyerTransformationError):
    """Report that an Actor response has metadata but no embedded dataset rows."""


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


def _parse_flipp_datetime(value: object) -> datetime | None:
    """Parse an ISO timestamp while leaving date-only and malformed values alone."""
    if not isinstance(value, str) or "T" not in value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalize_flipp_validity_window(mapped: dict[str, object]) -> None:
    """Convert Flipp's paired UTC boundary timestamps into inclusive dates.

    Flipp represents a local flyer week as UTC instants. For example, a Toronto
    offer may start at 04:00 UTC and end six days plus 23:59:59 later. Deriving
    the inclusive end date from that duration avoids incorrectly treating the UTC
    end timestamp's next-day date as locally valid.

    Date-only strings remain untouched for Pydantic. An isolated timestamp cannot
    safely establish the provider's local window, so it also remains untouched and
    fails clearly instead of silently producing a potentially incorrect date.
    """
    raw_start = mapped.get("validFrom")
    raw_end = mapped.get("validUntil")
    start = _parse_flipp_datetime(raw_start)
    end = _parse_flipp_datetime(raw_end)

    if start is None or end is None:
        return
    if (start.tzinfo is None) != (end.tzinfo is None):
        # Mixed aware/naive timestamps cannot be compared safely. Leave them for
        # Pydantic to reject rather than inventing a timezone assumption.
        return

    start_date: date = start.date()
    if end >= start:
        # timedelta.days floors the final partial day. That converts a range ending
        # at 03:59:59 UTC on the next date into the correct inclusive local date.
        # Subtracting one microsecond also handles an exclusive exact-midnight end.
        duration = end - start
        inclusive_duration = max(
            duration - timedelta(microseconds=1),
            timedelta(0),
        )
        end_date = start_date + timedelta(days=inclusive_duration.days)
    else:
        # Preserve a contradictory order so ProductOffer rejects it deterministically.
        end_date = end.date()

    mapped["validFrom"] = start_date
    mapped["validUntil"] = end_date


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

    _normalize_flipp_validity_window(mapped)

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
        raise MissingFlyerDatasetItemsError(
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
    """Run one Actor search, fetch its bounded dataset output, and validate it."""
    resolved_client = client or create_apify_mcp_client(handle_tool_errors=False)
    resolved_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else get_settings().apify_mcp_tool_timeout_seconds
    )
    raw_result = await search_raw_flyer_offers(
        request,
        client=resolved_client,
        timeout_seconds=resolved_timeout,
    )

    records_missing = False
    try:
        initial_offers = transform_apify_flyer_result(raw_result)
    except MissingFlyerDatasetItemsError:
        records_missing = True
        initial_offers = []

    if initial_offers:
        return initial_offers

    # An empty embedded sequence means only that this response carried no rows.
    # Actor metadata can still point to the dataset where Apify stored them.
    try:
        dataset_id = extract_default_dataset_id(raw_result.raw_response)
    except ApifyDatasetReadError:
        if records_missing:
            # Preserve the existing clear failure for an unstructured response;
            # only an explicitly present empty sequence represents zero offers.
            raise
        # A genuinely empty direct sequence without dataset metadata is valid.
        return []

    dataset_response = await fetch_apify_dataset_items(
        dataset_id,
        limit=request.max_items,
        client=resolved_client,
        timeout_seconds=resolved_timeout,
    )
    dataset_result = RawFlyerSearchResult(
        tool_name=raw_result.tool_name,
        request=request,
        raw_response=dataset_response,
    )
    return transform_apify_flyer_result(dataset_result)
