"""Kalshi related-markets lookup.

Two search modes, used together:

1. **series_tickers** — server-side filter via Kalshi's ``series_ticker`` param.
   Returns *all* markets under a series, which is exactly the "related markets"
   signal we want. Cheap and high-signal. We also auto-extract a series ticker
   from ``event.event_ticker`` (Kalshi tickers follow ``SERIES-YYYYMMDD-...``).

2. **keywords** — client-side fallback. Pages through open markets and matches
   on title/ticker. Slower; use only when the series isn't known.

We deliberately exclude the event's own ticker — using it would just give us
the benchmark price and yield zero edge.
"""
from __future__ import annotations

import logging
import re

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register

log = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _series_from_ticker(event_ticker: str) -> str | None:
    """Kalshi tickers look like ``SERIES-DATEFRAGMENT-...``. Return SERIES."""
    if not event_ticker:
        return None
    head = event_ticker.split("-", 1)[0]
    if re.fullmatch(r"[A-Z][A-Z0-9]+", head):
        return head
    return None


async def _fetch_series_markets(
    client: httpx.AsyncClient, series_ticker: str, limit: int = 100
) -> list[dict]:
    r = await client.get(
        f"{KALSHI_BASE}/markets",
        params={"series_ticker": series_ticker, "status": "open", "limit": limit},
    )
    r.raise_for_status()
    return r.json().get("markets", [])


async def _paginate_open_markets(
    client: httpx.AsyncClient, max_markets: int = 500
) -> list[dict]:
    out: list[dict] = []
    cursor: str | None = None
    while len(out) < max_markets:
        params: dict = {"status": "open", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        r = await client.get(f"{KALSHI_BASE}/markets", params=params)
        r.raise_for_status()
        payload = r.json()
        out.extend(payload.get("markets", []))
        cursor = payload.get("cursor")
        if not cursor:
            break
    return out[:max_markets]


def _summarize_market(m: dict, keyword: str | None = None) -> dict:
    return {
        "keyword": keyword,
        "ticker": m.get("ticker", ""),
        "event_ticker": m.get("event_ticker", ""),
        "title": m.get("title", ""),
        "yes_price": m.get("yes_ask") or m.get("last_price"),
        "no_price": m.get("no_ask"),
        "volume": m.get("volume", 0),
        "open_interest": m.get("open_interest", 0),
        "close_time": m.get("close_time", ""),
    }


@register(
    name="kalshi_related",
    description=(
        "Find related (non-identical) Kalshi prediction markets. Two modes:\n"
        "  - series_tickers: server-side filter (best when you know the series, "
        "e.g. KXCBDECISION for all central-bank rate decisions)\n"
        "  - keywords: text fallback (slower; pages through open markets)\n"
        "Returns YES prices and volumes — use as correlated base rates."
    ),
    args_schema=(
        '{"series_tickers": ["KXCBDECISION", "KXFED"], '
        '"keywords": ["bank of japan", "boj"], '
        '"limit_per_source": 10}'
    ),
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    series_tickers: list[str] = args.get("series_tickers") or []
    keywords: list[str] = args.get("keywords") or []
    limit_per_source = int(args.get("limit_per_source", 10))

    auto_series = _series_from_ticker(event.event_ticker)
    if auto_series and auto_series not in series_tickers:
        series_tickers.insert(0, auto_series)

    if not series_tickers and not keywords:
        return ResearchResult(
            tool="kalshi_related",
            summary="no series_tickers or keywords; skipping",
        )

    exclude_event = event.event_ticker
    items: list[dict] = []
    notes: list[str] = []

    async with httpx.AsyncClient(timeout=min(timeout_s, 30.0)) as client:
        for series in series_tickers[:5]:
            try:
                markets = await _fetch_series_markets(client, series, limit=100)
            except httpx.HTTPError as exc:
                notes.append(f"series {series}: {exc}")
                continue
            kept = 0
            for m in markets:
                if m.get("event_ticker") == exclude_event:
                    continue
                items.append(_summarize_market(m, keyword=f"series:{series}"))
                kept += 1
                if kept >= limit_per_source:
                    break
            notes.append(f"series {series}: {kept} markets")

        if keywords:
            try:
                pool = await _paginate_open_markets(client, max_markets=500)
            except httpx.HTTPError as exc:
                notes.append(f"pagination failed: {exc}")
                pool = []
            for kw in keywords[:5]:
                kw_lower = kw.lower()
                kept = 0
                for m in pool:
                    if m.get("event_ticker") == exclude_event:
                        continue
                    title = (m.get("title") or "").lower()
                    ticker = (m.get("ticker") or "").lower()
                    if kw_lower not in title and kw_lower not in ticker:
                        continue
                    items.append(_summarize_market(m, keyword=kw))
                    kept += 1
                    if kept >= limit_per_source:
                        break
                notes.append(f"keyword {kw!r}: {kept} markets")

    seen: set[str] = set()
    deduped: list[dict] = []
    for it in items:
        t = it["ticker"]
        if t and t not in seen:
            seen.add(t)
            deduped.append(it)

    summary = f"{len(deduped)} unique related markets (excluded self={exclude_event}). " + " | ".join(notes)
    return ResearchResult(tool="kalshi_related", items=deduped, summary=summary)
