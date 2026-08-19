"""Run exactly one minimal, paid Flipp Actor search for manual verification.

The script requires an explicit confirmation flag, requests one flyer record, and
contains no loop or retry. A client-side timeout does not guarantee that Apify has
stopped an already-started Actor, so inspect Apify Console before retrying a timeout.
"""

import argparse
import asyncio
from pprint import pprint

from pydantic import ValidationError

from app.core.logging import configure_logging
from app.core.mcp import MCPConfigurationError
from app.schemas.flyer_search import RawFlyerSearchRequest
from app.services.raw_flyer_search import (
    RawFlyerSearchError,
    search_raw_flyer_offers,
)


def parse_args() -> argparse.Namespace:
    """Parse one deliberately small manual search from the command line."""
    parser = argparse.ArgumentParser(
        description="Run one paid Apify Flipp search with maxItems fixed at 1."
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
        help="Confirm that one cost-sensitive Actor invocation is authorized.",
    )
    return parser.parse_args()


async def main() -> None:
    """Execute one confirmed query and print the untouched MCP response."""
    args = parse_args()
    if not args.confirm_paid_run:
        raise SystemExit(
            "No Actor was called. Add --confirm-paid-run to authorize exactly "
            "one paid invocation."
        )

    configure_logging()

    try:
        request = RawFlyerSearchRequest(
            query=args.query,
            postal_code=args.postal_code,
            max_items=1,
        )
        result = await search_raw_flyer_offers(request)
    except (ValidationError, MCPConfigurationError, RawFlyerSearchError) as exc:
        raise SystemExit(f"Raw flyer search failed: {exc}") from None

    print("One raw flyer search succeeded. Unmodified MCP response:")
    pprint(result.raw_response, sort_dicts=False)


if __name__ == "__main__":
    asyncio.run(main())
