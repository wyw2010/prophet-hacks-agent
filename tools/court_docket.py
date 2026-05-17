"""CourtListener REST API for US federal court case search.

Requires a free API token from https://www.courtlistener.com/help/api/rest/ —
they now require auth for the search endpoint (was previously public). Free
tier is 5000 requests/day.

Useful for any event involving specific litigation, judicial rulings,
scheduled hearings, sentencing dates, or government cases.

Endpoint types:
  o  — opinions (issued rulings)
  d  — dockets (case filings)
  r  — RECAP archive (PACER docs uploaded to CourtListener)
  oa — oral arguments
"""
from __future__ import annotations

import os

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register


COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v3"


@register(
    name="court_docket",
    description=(
        "US federal court case search via CourtListener. Use for events about "
        "specific lawsuits, rulings, convictions, sentencing, scheduled "
        "hearings, or government cases. Pass a free-text `query` and a "
        "`type`: 'o' for opinions (most useful for resolved-case research), "
        "'d' for dockets (active case filings), 'r' for RECAP archive."
    ),
    args_schema=(
        '{"query": "Trump immunity ruling", "type": "o", "limit": 10}'
    ),
    requires_env=["COURTLISTENER_API_KEY"],
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    token = os.environ["COURTLISTENER_API_KEY"]
    query = args.get("query") or event.title
    result_type = args.get("type", "o")
    limit = max(1, min(int(args.get("limit", 10)), 25))

    async with httpx.AsyncClient(
        timeout=min(timeout_s, 20.0),
        headers={
            "User-Agent": "prophet-hacks-agent/0.1 (https://github.com/wyw2010/prophet-hacks-agent)",
            "Authorization": f"Token {token}",
        },
    ) as client:
        try:
            r = await client.get(
                f"{COURTLISTENER_BASE}/search/",
                params={"q": query, "type": result_type, "order_by": "score desc"},
            )
            r.raise_for_status()
            payload = r.json()
        except httpx.HTTPError as exc:
            return ResearchResult(tool="court_docket", error=f"http error: {exc}")

    raw = payload.get("results") or []
    items: list[dict] = []
    for hit in raw[:limit]:
        items.append(
            {
                "case_name": hit.get("caseName") or hit.get("case_name", ""),
                "court": hit.get("court") or hit.get("court_id", ""),
                "date_filed": hit.get("dateFiled") or hit.get("date_filed", ""),
                "date_argued": hit.get("dateArgued", ""),
                "docket_number": hit.get("docketNumber") or hit.get("docket_number", ""),
                "judge": hit.get("judge", ""),
                "snippet": (hit.get("snippet") or "")[:500],
                "url": f"https://www.courtlistener.com{hit.get('absolute_url', '')}"
                if hit.get("absolute_url") else hit.get("download_url", ""),
            }
        )

    summary = (
        f"Found {len(items)} court results for {query!r} "
        f"(type={result_type}, total available={payload.get('count', 'unknown')})."
    )
    return ResearchResult(tool="court_docket", items=items, summary=summary)
