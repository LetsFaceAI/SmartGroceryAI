"""Apify Flipp implementation of the provider-neutral grocery search contract.

Only this integration layer translates shared search requests into Actor inputs.
MCP discovery, LangChain tools, raw responses, and dataset fallback remain behind
the existing Apify services and never cross the ``GrocerySearchProvider`` boundary.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from pydantic import ValidationError

from app.core.logging import get_logger
from app.providers.base import (
    GrocerySearchProvider,
    SearchProviderExecutionError,
    SearchProviderTimeoutError,
    UnsupportedSearchIntentError,
)
from app.schemas.flyer_search import MAX_RAW_SEARCH_ITEMS, RawFlyerSearchRequest
from app.schemas.product_offer import ProductOffer
from app.schemas.search_provider import (
    GrocerySearchIntent,
    GrocerySearchRequest,
    GrocerySearchResult,
    SearchProvenance,
)
from app.schemas.shopping import ConstraintRequirement
from app.services.apify_dataset_reader import (
    ApifyDatasetReadError,
    ApifyDatasetReadTimeoutError,
)
from app.services.apify_flyer_transformer import (
    ApifyFlyerTransformationError,
    search_product_offers,
)
from app.services.raw_flyer_search import (
    RawFlyerSearchError,
    RawFlyerSearchTimeoutError,
)
from app.services.search_request_policy import (
    ExternalActorBudgetExceededError,
    ExternalActorCallBudget,
)

logger = get_logger(__name__)


class _OfferSearch(Protocol):
    """Describe the private validated Apify pipeline dependency used in tests."""

    async def __call__(
        self,
        request: RawFlyerSearchRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> list[ProductOffer]: ...


def _utc_now() -> datetime:
    """Return an unambiguous retrieval timestamp for result provenance."""
    return datetime.now(UTC)


def _build_flipp_query(request: GrocerySearchRequest) -> str:
    """Build one deterministic Actor query without dropping item constraints.

    Flipp accepts one string query. Required constraint phrases are included once,
    followed by the requested item name. Optional constraints, quantity, and
    free-form notes are intentionally not used to narrow discovery because doing
    so could exclude useful candidates. The complete ``ShoppingItem`` remains on
    the shared result for deterministic matching after retrieval.
    """
    candidate_parts = [
        constraint.value
        for constraint in request.item.constraints
        if constraint.requirement is ConstraintRequirement.REQUIRED
    ]
    candidate_parts.append(request.item.name)
    query_parts: list[str] = []
    seen: set[str] = set()
    for value in candidate_parts:
        normalized = " ".join(value.casefold().split())
        if normalized not in seen:
            query_parts.append(value)
            seen.add(normalized)
    return " ".join(query_parts)


class ApifyFlippProvider(GrocerySearchProvider):
    """Search Flipp flyer deals through one cost-bounded Apify Actor pipeline.

    A call budget is mandatory rather than optional. Future orchestration may own
    and share that budget across multiple provider calls, while this provider makes
    it impossible to invoke the paid pipeline without consuming a budget slot.
    """

    provider_name = "apify_flipp"

    def __init__(
        self,
        *,
        call_budget: ExternalActorCallBudget,
        timeout_seconds: float | None = None,
        offer_search: _OfferSearch = search_product_offers,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._call_budget = call_budget
        self._timeout_seconds = timeout_seconds
        self._offer_search = offer_search
        self._clock = clock

    async def search(self, request: GrocerySearchRequest) -> GrocerySearchResult:
        """Run one flyer search and return only shared, validated domain models."""
        if request.intent is not GrocerySearchIntent.FLYER_DEALS:
            raise UnsupportedSearchIntentError(
                "ApifyFlippProvider supports flyer-deal searches only."
            )

        try:
            raw_request = RawFlyerSearchRequest(
                query=_build_flipp_query(request),
                postal_code=request.postal_code,
                merchant_name=request.store,
                # The provider-neutral cap can exceed the deliberately small paid
                # Actor cap. Returning fewer results is safe; raising the Actor cap
                # is not permitted from a caller-controlled shared request.
                max_items=min(request.max_results, MAX_RAW_SEARCH_ITEMS),
            )
        except ValidationError as exc:
            # Do not expose the rejected query text. The common request may be
            # valid while exceeding a stricter provider-specific input limit.
            raise SearchProviderExecutionError(
                "The grocery search request cannot be represented safely by "
                "the Apify Flipp provider."
            ) from exc

        try:
            # The existing budget tracks paid Actor attempts and active calls. Its
            # slot remains consumed after failures because a timed-out Actor may
            # still be running. Dataset fallback is read-only and stays inside the
            # single validated pipeline invocation below.
            with self._call_budget.actor_call():
                offers = await self._offer_search(
                    raw_request,
                    timeout_seconds=self._timeout_seconds,
                )
        except (RawFlyerSearchTimeoutError, ApifyDatasetReadTimeoutError) as exc:
            raise SearchProviderTimeoutError(
                "The Apify Flipp search timed out and was not retried."
            ) from exc
        except (
            RawFlyerSearchError,
            ApifyDatasetReadError,
            ApifyFlyerTransformationError,
        ) as exc:
            raise SearchProviderExecutionError(
                "The Apify Flipp search could not produce validated offers."
            ) from exc
        except ExternalActorBudgetExceededError:
            # Budget exhaustion is application policy, not an Apify transport
            # failure. Preserve it so orchestration can stop all further calls.
            raise
        except Exception as exc:
            # External exceptions can contain request URLs or authorization data.
            # Log only the public provider name and exception class.
            logger.error(
                "Grocery search provider failed provider=%s error_type=%s",
                self.provider_name,
                type(exc).__name__,
            )
            raise SearchProviderExecutionError(
                "The Apify Flipp provider failed unexpectedly."
            ) from exc

        return GrocerySearchResult(
            request=request,
            offers=tuple(offers),
            provenance=SearchProvenance(
                provider_name=self.provider_name,
                retrieved_at=self._clock(),
            ),
        )
