"""Kalshi related-markets lookup.

Searches Kalshi's public markets endpoint for tickers matching given keywords,
*excluding* the event's own ticker. Returns each related market's current YES
price and recent price trend as a cross-market base-rate signal.

We deliberately do NOT query the same market_ticker — that would just give us
the benchmark price and yield zero edge.
"""
from __future__ import annotations

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register


KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


@register(
    name="kalshi_related",
    description=(
        "Find related (non-identical) Kalshi prediction markets by keyword. "
        "Returns current YES prices and volumes — use as correlated base rates."
    ),
    args_schema='{"keywords": ["nba playoffs", "cleveland"], "limit_per_keyword": 5}',
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    keywords: list[str] = args.get("keywords") or []
    limit = int(args.get("limit_per_keyword", 5))
    if not keywords:
        return ResearchResult(
            tool="kalshi_related", summary="no keywords supplied; skipping"
        )

    exclude_event = event.event_ticker
    items: list[dict] = []

    async with httpx.AsyncClient(timeout=min(timeout_s, 30.0)) as client:
        for kw in keywords[:5]:
            try:
                r = await client.get(
                    f"{KALSHI_BASE}/markets",
                    params={"limit": 50, "status": "open"},
                )
                r.raise_for_status()
                payload = r.json()
            except httpx.HTTPError as exc:
                return ResearchResult(
                    tool="kalshi_related",
                    items=items,
                    error=f"http error: {exc}",
                )

            kw_lower = kw.lower()
            for m in payload.get("markets", []):
                title = (m.get("title") or "").lower()
                ticker = m.get("ticker", "")
                event_ticker = m.get("event_ticker", "")
                if event_ticker == exclude_event:
                    continue
                if kw_lower not in title and kw_lower not in ticker.lower():
                    continue
                items.append(
                    {
                        "keyword": kw,
                        "ticker": ticker,
                        "event_ticker": event_ticker,
                        "title": m.get("title", ""),
                        "yes_price": m.get("yes_ask") or m.get("last_price"),
                        "volume": m.get("volume", 0),
                        "close_time": m.get("close_time", ""),
                    }
                )
                if sum(1 for x in items if x["keyword"] == kw) >= limit:
                    break

    summary = (
        f"Found {len(items)} related Kalshi markets across {len(keywords)} keyword(s); "
        f"excluded self-ticker {exclude_event}."
    )
    return ResearchResult(tool="kalshi_related", items=items, summary=summary)
