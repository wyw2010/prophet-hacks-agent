"""Polymarket related-markets lookup via Gamma API.

Cross-market signal from a second prediction-market venue with a different
participant base than Kalshi.
"""
from __future__ import annotations

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register


GAMMA_BASE = "https://gamma-api.polymarket.com"


@register(
    name="polymarket_related",
    description=(
        "Find related Polymarket markets by keyword. Returns current YES prices "
        "and volumes — different participant base than Kalshi, valuable diversity."
    ),
    args_schema='{"keywords": ["election", "fed rate"], "limit_per_keyword": 5}',
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    keywords: list[str] = args.get("keywords") or []
    limit = int(args.get("limit_per_keyword", 5))
    if not keywords:
        return ResearchResult(
            tool="polymarket_related", summary="no keywords supplied; skipping"
        )

    items: list[dict] = []
    async with httpx.AsyncClient(timeout=min(timeout_s, 30.0)) as client:
        for kw in keywords[:5]:
            try:
                r = await client.get(
                    f"{GAMMA_BASE}/markets",
                    params={"active": "true", "closed": "false", "limit": 50},
                )
                r.raise_for_status()
                payload = r.json()
            except httpx.HTTPError as exc:
                return ResearchResult(
                    tool="polymarket_related",
                    items=items,
                    error=f"http error: {exc}",
                )

            kw_lower = kw.lower()
            markets = payload if isinstance(payload, list) else payload.get("data", [])
            for m in markets:
                question = (m.get("question") or "").lower()
                if kw_lower not in question:
                    continue
                items.append(
                    {
                        "keyword": kw,
                        "question": m.get("question", ""),
                        "slug": m.get("slug", ""),
                        "outcome_prices": m.get("outcomePrices"),
                        "volume": m.get("volume", 0),
                        "end_date": m.get("endDate", ""),
                    }
                )
                if sum(1 for x in items if x["keyword"] == kw) >= limit:
                    break

    summary = f"Found {len(items)} related Polymarket markets across {len(keywords)} keyword(s)."
    return ResearchResult(tool="polymarket_related", items=items, summary=summary)
