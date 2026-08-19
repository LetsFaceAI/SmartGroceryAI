"""Run one opt-in, end-to-end smoke test of the live Apify flyer pipeline.

This script is intentionally outside pytest discovery because invoking the Flipp
Actor may consume Apify credit. It has no retry or loop, fixes ``max_items`` at
one, and requires an explicit confirmation flag before any connection is opened.
"""

import argparse
import asyncio

from pydantic import ValidationError

from app.core.logging import configure_logging
from app.core.mcp import MCPConfigurationError
from app.schemas.flyer_search import RawFlyerSearchRequest
from app.schemas.product_offer import ProductOffer
from app.services.apify_dataset_reader import ApifyDatasetReadError
from app.services.apify_flyer_transformer import (
    ApifyFlyerTransformationError,
    search_product_offers,
)
from app.services.raw_flyer_search import RawFlyerSearchError


def parse_args() -> argparse.Namespace:
    """Collect one minimal query and require explicit paid-run authorization."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one paid Apify Actor invocation and validate at most one offer."
        )
    )
    parser.add_argument("--query", required=True, help="One grocery keyword.")
    parser.add_argument(
        "--postal-code",
        required=True,
        help="US ZIP code or Canadian postal code.",
    )
    parser.add_argument(
        "--confirm-paid-run",
        action="store_true",
        help="Authorize exactly one cost-sensitive Flipp Actor invocation.",
    )
    return parser.parse_args()


def print_offer_summary(offer: ProductOffer) -> None:
    """Print selected validated fields without raw responses or credentials."""
    validity = "validity unavailable"
    if offer.valid_from is not None or offer.valid_until is not None:
        valid_from = offer.valid_from.isoformat() if offer.valid_from else "unknown"
        valid_until = offer.valid_until.isoformat() if offer.valid_until else "unknown"
        validity = f"valid {valid_from} to {valid_until}"

    print(
        f"- {offer.product_name} | {offer.store} | "
        f"{offer.currency} {offer.price} | {offer.promotion_status.value} | {validity}"
    )


async def main() -> None:
    """Execute the production MCP pipeline once and display a safe result summary."""
    args = parse_args()
    if not args.confirm_paid_run:
        raise SystemExit(
            "No MCP connection or Actor call was made. Add --confirm-paid-run to "
            "authorize exactly one paid invocation."
        )

    try:
        request = RawFlyerSearchRequest(
            query=args.query,
            postal_code=args.postal_code,
            max_items=1,
        )
    except ValidationError as exc:
        # These errors contain only command-line query data, not MCP credentials.
        raise SystemExit(f"Invalid smoke-test input: {exc}") from None

    # The smoke command promises a concise validated summary. WARNING retains
    # actionable failures while suppressing verbose HTTP and MCP schema logs.
    configure_logging(level="WARNING")

    try:
        offers = await search_product_offers(request)
    except MCPConfigurationError as exc:
        # The configuration exception gives safe, actionable missing-setting advice.
        raise SystemExit(f"MCP configuration failed: {exc}") from None
    except RawFlyerSearchError as exc:
        raise SystemExit(f"Live flyer search failed: {exc}") from None
    except ApifyDatasetReadError as exc:
        raise SystemExit(f"Flyer dataset retrieval failed: {exc}") from None
    except ApifyFlyerTransformationError as exc:
        raise SystemExit(f"Flyer response validation failed: {exc}") from None
    except ValidationError:
        # Avoid echoing environment values from a settings-validation failure.
        raise SystemExit(
            "Application configuration validation failed. Check the MCP values in .env."
        ) from None

    if not offers:
        raise SystemExit(
            "MCP connection and Actor invocation succeeded, but no offer was returned; "
            "the ProductOffer transformation smoke check is incomplete."
        )

    print("Apify MCP flyer pipeline smoke test succeeded.")
    print(f"Validated offers: {len(offers)}")
    for offer in offers:
        print_offer_summary(offer)


if __name__ == "__main__":
    asyncio.run(main())
