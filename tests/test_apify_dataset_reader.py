"""Offline tests for the read-only Apify dataset follow-up boundary."""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.services.apify_dataset_reader import (
    APIFY_DATASET_ITEMS_TOOL_NAME,
    ApifyDatasetReadError,
    extract_default_dataset_id,
    fetch_apify_dataset_items,
)

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "apify"


def load_fixture(filename: str) -> dict[str, Any]:
    """Read saved Actor metadata without contacting Apify."""
    return cast(
        dict[str, Any],
        json.loads((FIXTURE_DIRECTORY / filename).read_text(encoding="utf-8")),
    )


def make_client() -> Mock:
    """Return a client-shaped mock that cannot open a real MCP connection."""
    return Mock(spec=MultiServerMCPClient)


def test_extract_default_dataset_id_from_actor_artifact() -> None:
    """The current nested Apify storage contract should yield its primary ID."""
    fixture = load_fixture("pipeline_actor_metadata_response.json")
    response = ToolMessage(
        content=fixture["content"],
        artifact=fixture["artifact"],
        tool_call_id="fixture-actor-call",
        name="crawlerbros--flipp-grocery-deals-scraper",
    )

    assert extract_default_dataset_id(response) == "fixture-dataset-id"


def test_extract_default_dataset_id_rejects_missing_metadata() -> None:
    """A summary without rows or a dataset ID must produce actionable failure."""
    with pytest.raises(ApifyDatasetReadError, match="default dataset ID"):
        extract_default_dataset_id({"content": "SUCCEEDED. 1 item."})


@pytest.mark.anyio
async def test_fetch_dataset_items_invokes_read_only_tool_once() -> None:
    """The helper call should be bounded and preserve its structured artifact."""
    response: dict[str, Any] = {"artifact": {"structured_content": {"items": []}}}
    dataset_tool = Mock(spec=BaseTool)
    dataset_tool.name = APIFY_DATASET_ITEMS_TOOL_NAME
    dataset_tool.ainvoke = AsyncMock(return_value=response)

    with patch(
        "app.services.apify_dataset_reader.discover_apify_mcp_tools",
        new=AsyncMock(return_value=[dataset_tool]),
    ):
        result = await fetch_apify_dataset_items(
            "fixture-dataset-id",
            limit=1,
            client=make_client(),
            timeout_seconds=1,
        )

    assert result is response
    tool_call = dataset_tool.ainvoke.await_args.args[0]
    assert tool_call["type"] == "tool_call"
    assert tool_call["name"] == APIFY_DATASET_ITEMS_TOOL_NAME
    assert tool_call["args"] == {
        "datasetId": "fixture-dataset-id",
        "limit": 1,
        "clean": True,
    }


@pytest.mark.anyio
async def test_fetch_dataset_items_rejects_unbounded_limit_before_discovery() -> None:
    """Direct callers cannot bypass the application's five-item safety ceiling."""
    with pytest.raises(ValueError, match="between 1 and 5"):
        await fetch_apify_dataset_items(
            "fixture-dataset-id",
            limit=6,
            client=make_client(),
            timeout_seconds=1,
        )
