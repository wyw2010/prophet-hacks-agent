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

import asyncio
import logging
import re

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from schemas import EventRequest, ResearchResult
from tools.base import register

log = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


class _KalshiRateLimited(Exception):
    """Raised on HTTP 429 so tenacity can retry with exponential backoff."""


def _series_from_ticker(event_ticker: str) -> str | None:
    """Kalshi tickers look like ``SERIES-DATEFRAGMENT-...``. Return SERIES."""
    if not event_ticker:
        return None
    head = event_ticker.split("-", 1)[0]
    if re.fullmatch(r"[A-Z][A-Z0-9]+", head):
        return head
    return None


async def _kalshi_get(client: httpx.AsyncClient, params: dict) -> dict:
    """GET /markets with retry+backoff on 429.

    Backoff: 2s, 4s, 8s (capped at 10s). Gives up after 4 attempts and
    re-raises so the caller surfaces the failure.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(_KalshiRateLimited),
        reraise=True,
    ):
        with attempt:
            r = await client.get(f"{KALSHI_BASE}/markets", params=params)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After", "unset")
                log.warning("Kalshi 429 (Retry-After=%s); will back off", retry_after)
                raise _KalshiRateLimited(retry_after)
            r.raise_for_status()
            return r.json()
    return {}  # unreachable; satisfies type checker


async def _fetch_series_markets(
    client: httpx.AsyncClient, series_ticker: str, limit: int = 100
) -> list[dict]:
    payload = await _kalshi_get(
        client,
        params={"series_ticker": series_ticker, "status": "open", "limit": limit},
    )
    return payload.get("markets", [])


async def _paginate_open_markets(
    client: httpx.AsyncClient, max_markets: int = 200
) -> list[dict]:
    """Page through open markets. Default depth dropped from 500 to 200 — five
    paginated GETs back-to-back kept tripping the rate limit and the additional
    300 markets rarely contained matches anyway.
    """
    out: list[dict] = []
    cursor: str | None = None
    while len(out) < max_markets:
        params: dict = {"status": "open", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        payload = await _kalshi_get(client, params=params)
        out.extend(payload.get("markets", []))
        cursor = payload.get("cursor")
        if not cursor:
            break
        # Tiny inter-page pause to be polite under rate limits.
        await asyncio.sleep(0.2)
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


def _has_signal(m: dict) -> bool:
    """Skip markets that have no useful price OR volume signal — they're noise.

    Common case: brand-new series markets where Kalshi has listed the contract
    but no one's traded yet (volume=0, yes_ask=null). Returning these to the
    forecaster just dilutes the brief.
    """
    yes_price = m.get("yes_ask") if m.get("yes_ask") is not None else m.get("last_price")
    has_price = yes_price is not None
    has_volume = (m.get("volume") or 0) > 0
    has_oi = (m.get("open_interest") or 0) > 0
    return has_price or has_volume or has_oi


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
            skipped_no_signal = 0
            for m in markets:
                if m.get("event_ticker") == exclude_event:
                    continue
                if not _has_signal(m):
                    skipped_no_signal += 1
                    continue
                items.append(_summarize_market(m, keyword=f"series:{series}"))
                kept += 1
                if kept >= limit_per_source:
                    break
            notes.append(
                f"series {series}: {kept} markets (skipped {skipped_no_signal} no-signal)"
            )

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
                    if not _has_signal(m):
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
