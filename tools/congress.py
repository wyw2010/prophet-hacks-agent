"""Congress.gov bills lookup.

Args:
    query: free-text search across bill titles/summaries
    congress: optional congress number (default: 119 for 2025-2027)
"""
from __future__ import annotations

import os

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register


CONGRESS_BASE = "https://api.congress.gov/v3"


@register(
    name="congress_bills",
    description=(
        "Congress.gov API for federal legislation. Use for events about specific "
        "bills passing, vote outcomes, or chamber actions."
    ),
    args_schema='{"query": "appropriations 2026", "congress": 119, "limit": 10}',
    requires_env=["CONGRESS_API_KEY"],
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    key = os.environ["CONGRESS_API_KEY"]
    query = args.get("query", event.title)
    congress = int(args.get("congress", 119))
    limit = int(args.get("limit", 10))

    async with httpx.AsyncClient(timeout=min(timeout_s, 20.0)) as client:
        try:
            r = await client.get(
                f"{CONGRESS_BASE}/bill/{congress}",
                params={
                    "api_key": key,
                    "format": "json",
                    "limit": limit,
                    "q": query,
                },
            )
            r.raise_for_status()
            payload = r.json()
        except httpx.HTTPError as exc:
            return ResearchResult(tool="congress_bills", error=f"http error: {exc}")

    items = []
    for bill in payload.get("bills", [])[:limit]:
        items.append(
            {
                "number": bill.get("number"),
                "type": bill.get("type"),
                "title": bill.get("title"),
                "latest_action": bill.get("latestAction", {}).get("text", ""),
                "action_date": bill.get("latestAction", {}).get("actionDate", ""),
                "url": bill.get("url", ""),
            }
        )

    summary = f"Found {len(items)} bills matching {query!r} in Congress {congress}."
    return ResearchResult(tool="congress_bills", items=items, summary=summary)
