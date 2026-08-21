"""Run one opt-in live test of the complete grocery-agent workflow."""

import argparse
import asyncio

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from app.agents.grocery_coordinator import (
    create_request_scoped_grocery_coordinator,
)
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.schemas.search_provider import GrocerySearchRequest
from app.schemas.shopping import ShoppingItem, ShoppingRequest

MAX_AGENT_RECURSION_LIMIT = 6


def parse_args() -> argparse.Namespace:
    """Collect one grocery item and require explicit paid-run approval."""
    parser = argparse.ArgumentParser(
        description=("Run one bounded OpenAI and Apify grocery-agent interaction.")
    )
    parser.add_argument(
        "--item",
        required=True,
        help="One grocery item to search for.",
    )
    parser.add_argument(
        "--postal-code",
        required=True,
        help="Canadian postal code or US ZIP code.",
    )
    parser.add_argument(
        "--confirm-paid-run",
        action="store_true",
        help="Authorize bounded OpenAI usage and one Apify Actor attempt.",
    )
    return parser.parse_args()


async def main() -> None:
    """Execute one complete production agent interaction."""
    args = parse_args()
    if not args.confirm_paid_run:
        raise SystemExit(
            "No model or Actor call was made. Add --confirm-paid-run to "
            "authorize the bounded live test."
        )

    try:
        settings = get_settings()
        validated_search = GrocerySearchRequest(
            item=ShoppingItem(name=args.item),
            postal_code=args.postal_code,
        )
        shopping_request = ShoppingRequest(
            items=[validated_search.item],
        )
    except ValidationError:
        raise SystemExit(
            "Live-test input or application configuration is invalid."
        ) from None

    if (
        settings.openai_api_key is None
        or not settings.openai_api_key.get_secret_value().strip()
    ):
        raise SystemExit("OPENAI_API_KEY is not configured.")

    if settings.apify_mcp_server_url is None:
        raise SystemExit("APIFY_MCP_SERVER_URL is not configured.")

    if (
        settings.apify_api_token is None
        or not settings.apify_api_token.get_secret_value().strip()
    ):
        raise SystemExit("APIFY_API_TOKEN is not configured.")

    if settings.search_max_external_actor_calls_per_request != 1:
        raise SystemExit(
            "Set SEARCH_MAX_EXTERNAL_ACTOR_CALLS_PER_REQUEST=1 "
            "before running this smoke test."
        )

    if settings.search_max_concurrency != 1:
        raise SystemExit("Set SEARCH_MAX_CONCURRENCY=1 before running this smoke test.")

    configure_logging(level="WARNING")

    prompt = (
        f"Find the current cheapest flyer deal for "
        f"{validated_search.item.name} near postal code "
        f"{validated_search.postal_code}. "
        "Use find_flyer_deals exactly once. "
        "Base your answer only on the tool result."
    )

    try:
        agent = create_request_scoped_grocery_coordinator(
            shopping_request,
            settings=settings,
        )
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=prompt),
                ]
            },
            config={"recursion_limit": MAX_AGENT_RECURSION_LIMIT},
        )
    except GraphRecursionError:
        raise SystemExit(
            "The live agent exceeded its bounded recursion limit."
        ) from None
    except Exception as exc:
        raise SystemExit(
            "The live agent failed safely with error type "
            f"{type(exc).__name__}. No exception details were printed."
        ) from None

    messages = result["messages"]
    tool_messages = [
        message for message in messages if isinstance(message, ToolMessage)
    ]

    if len(tool_messages) != 1:
        raise SystemExit(
            "The live agent did not execute exactly one tool call; "
            f"observed {len(tool_messages)}."
        )

    final_message = messages[-1]
    if not isinstance(final_message, AIMessage):
        raise SystemExit("The live agent did not return a final AIMessage.")

    final_answer = final_message.text.strip()
    if not final_answer:
        raise SystemExit("The live agent returned an empty final answer.")

    model_responses = sum(isinstance(message, AIMessage) for message in messages)

    print("SmartGroceryAI live agent smoke test succeeded.")
    print("Tool calls executed: 1")
    print(f"Model responses: {model_responses}")
    print("Final answer:")
    print(final_answer)


if __name__ == "__main__":
    asyncio.run(main())
