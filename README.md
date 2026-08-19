# SmartGroceryAI

An AI-powered website for finding nearby grocery deals and building cost-effective shopping plans.

## Manual Apify pipeline smoke test

The live smoke test verifies the configured MCP connection, invokes the Flipp
Actor once, preserves its raw structured response, and validates at most one
result as a `ProductOffer`. It is a manual script rather than a pytest test, so CI
and `uv run pytest` never execute it.

The Actor response may contain only run metadata. In that case, the pipeline uses
the returned default dataset ID to call Apify's read-only `get-dataset-items` MCP
helper once with `limit=1`. This retrieves the existing run output and does not
start a second Actor run.

Before running it, configure `APIFY_MCP_SERVER_URL` and `APIFY_API_TOKEN` in the
local, untracked `.env` file. Because the Actor may consume Apify credit, the
script requires an explicit confirmation flag and never retries automatically:

```powershell
uv run python -m scripts.smoke_test_apify_pipeline --query "cantaloupe" --postal-code "YOUR_POSTAL_CODE" --confirm-paid-run
```

Successful output contains only a count and selected validated fields. It never
prints the raw MCP response, authorization header, or Apify token:

```text
Apify MCP flyer pipeline smoke test succeeded.
Validated offers: 1
- Whole Cantaloupe | Example Grocer | CAD 2.99 | sale | valid 2026-08-20 to 2026-08-26
```

If the query returns no flyer offer, the script reports that the connection and
Actor invocation succeeded but the transformation check remains incomplete. Use
a commonly advertised item for the single manual run rather than retrying in a
loop.
