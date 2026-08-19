"""Read the bounded output dataset referenced by an Apify Actor response.

Apify Actor tools return run and storage metadata rather than embedding every
dataset row. This service performs the documented follow-up read through the
MCP ``get-dataset-items`` helper without starting or retrying an Actor run.
"""

import asyncio
from collections.abc import Mapping
from typing import cast
from uuid import uuid4

from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.logging import get_logger
from app.schemas.flyer_search import MAX_RAW_SEARCH_ITEMS
from app.services.mcp_tool_discovery import (
    MCPToolDiscoveryError,
    discover_apify_mcp_tools,
)

APIFY_DATASET_ITEMS_TOOL_NAME = "get-dataset-items"

logger = get_logger(__name__)


class ApifyDatasetReadError(RuntimeError):
    """Report dataset metadata or retrieval failures without exposing credentials."""


class ApifyDatasetReadTimeoutError(ApifyDatasetReadError):
    """Report that the single read-only dataset request exceeded its deadline."""


def _find_dataset_id(value: object) -> str | None:
    """Find the documented default dataset ID inside MCP response wrappers."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _find_dataset_id(model_dump())

    if not isinstance(value, Mapping):
        return None

    mapping = cast(Mapping[str, object], value)
    storages = mapping.get("storages")
    if isinstance(storages, Mapping):
        datasets = storages.get("datasets")
        if isinstance(datasets, Mapping):
            default_dataset = datasets.get("default")
            if isinstance(default_dataset, Mapping):
                dataset_id = default_dataset.get("id")
                if isinstance(dataset_id, str) and dataset_id.strip():
                    return dataset_id.strip()

    # Older Apify response versions used these flatter names. Supporting them is
    # harmless and keeps saved responses usable while the nested path remains the
    # preferred current contract.
    for key in ("datasetId", "defaultDatasetId"):
        dataset_id = mapping.get(key)
        if isinstance(dataset_id, str) and dataset_id.strip():
            return dataset_id.strip()

    for key in ("artifact", "structured_content", "structuredContent"):
        if key in mapping:
            dataset_id = _find_dataset_id(mapping[key])
            if dataset_id is not None:
                return dataset_id
    return None


def extract_default_dataset_id(raw_response: object) -> str:
    """Return the Actor's primary dataset ID or fail before any dataset request."""
    dataset_id = _find_dataset_id(raw_response)
    if dataset_id is None:
        raise ApifyDatasetReadError(
            "The Actor response contained neither flyer items nor a default dataset ID."
        )
    return dataset_id


def _find_dataset_tool(tools: list[BaseTool]) -> BaseTool:
    """Select only Apify's read-only dataset helper from discovered MCP tools."""
    for tool in tools:
        if tool.name == APIFY_DATASET_ITEMS_TOOL_NAME:
            return tool
    raise ApifyDatasetReadError(
        f"Required MCP tool '{APIFY_DATASET_ITEMS_TOOL_NAME}' was not discovered."
    )


async def fetch_apify_dataset_items(
    dataset_id: str,
    *,
    limit: int,
    client: MultiServerMCPClient,
    timeout_seconds: float,
) -> object:
    """Fetch one bounded dataset page without invoking or retrying the Actor.

    The caller supplies the already-configured client used for the Actor pipeline.
    Automated tests inject mocks, so this function never performs network I/O in CI.
    """
    if not dataset_id.strip():
        raise ValueError("dataset_id must not be empty.")
    if not 1 <= limit <= MAX_RAW_SEARCH_ITEMS:
        raise ValueError(
            f"limit must be between 1 and {MAX_RAW_SEARCH_ITEMS} to keep the "
            "dataset read bounded."
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0.")

    try:
        async with asyncio.timeout(timeout_seconds):
            tools = await discover_apify_mcp_tools(client)
            dataset_tool = _find_dataset_tool(tools)
            tool_call = ToolCall(
                name=dataset_tool.name,
                args={
                    "datasetId": dataset_id,
                    "limit": limit,
                    "clean": True,
                },
                id=uuid4().hex,
                type="tool_call",
            )
            response = await dataset_tool.ainvoke(tool_call)
    except TimeoutError as exc:
        raise ApifyDatasetReadTimeoutError(
            "The read-only Apify dataset request timed out and was not retried."
        ) from exc
    except ApifyDatasetReadError:
        raise
    except MCPToolDiscoveryError as exc:
        raise ApifyDatasetReadError(
            "Could not discover the Apify dataset-items MCP tool."
        ) from exc
    except Exception as exc:
        logger.error(
            "Apify dataset read failed tool=%s error_type=%s",
            APIFY_DATASET_ITEMS_TOOL_NAME,
            type(exc).__name__,
        )
        raise ApifyDatasetReadError(
            "The Apify dataset-items MCP tool failed during its single read."
        ) from exc

    logger.info("Apify dataset items retrieved limit=%s", limit)
    return response
