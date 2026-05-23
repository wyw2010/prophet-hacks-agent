"""Kalshi related-markets lookup.

Two search modes, used together:

1. **series_tickers** — server-side filter via Kalshi's ``series_ticker`` param.
   Returns *all* markets under a series, which is exactly the "related markets"
   signal we want. Cheap and high-signal. We also auto-extract a series ticker
   from ``event.event_ticker`` (Kalshi tickers follow ``SERIES-YYYYMMDD-...``).

2. **keywords** — client-side fallback. Pages through open markets and matches
   on title/ticker. Slower; use only when the series isn't known.

By default we INCLUDE the event's own ticker — the live market price for the
exact question is the strongest calibration anchor available, and the
forecaster prompt explicitly asks for cross-market anchors. Pass
``include_self=false`` if you specifically want to exclude it.
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


async def _fetch_event_detail(
    client: httpx.AsyncClient, event_ticker: str
) -> dict | None:
    """Fetch the /events/{ticker} endpoint, which returns *full* market data
    (including live ``yes_bid_dollars`` / ``no_bid_dollars`` / ``last_price_dollars``).

    The /markets list endpoint we use elsewhere returns sparse data — these
    price fields are usually null there. /events is the only public endpoint
    that surfaces a usable price for our forecaster anchor.

    Returns None on any failure (including 429 after retries) so the caller
    can fall back gracefully.
    """
    if not event_ticker:
        return None
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            retry=retry_if_exception_type(_KalshiRateLimited),
            reraise=True,
        ):
            with attempt:
                r = await client.get(f"{KALSHI_BASE}/events/{event_ticker}")
                if r.status_code == 429:
                    raise _KalshiRateLimited(r.headers.get("Retry-After", "unset"))
                if r.status_code != 200:
                    return None
                return r.json()
    except Exception as exc:  # noqa: BLE001 — best effort; fail soft
        log.warning("event detail fetch failed for %s: %s", event_ticker, exc)
        return None
    return None


def _coerce_float(x):
    """Coerce to float, returning None on any failure or for null/empty."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _compute_yes_price(m: dict) -> tuple[float | None, str]:
    """Return (yes_price, source) using the best available signal from a
    Kalshi market dict (the full version returned by /events/{ticker}).

    Priority:
      1. Two-sided live quote: midpoint of (yes_bid, 1 - no_bid).
      2. One-sided yes_bid.
      3. Inverse of one-sided no_bid (i.e., 1 - no_bid).
      4. last_price (settled or last trade; biased toward winner if resolved).

    Returns (None, "none") if no usable signal.
    """
    yb = _coerce_float(m.get("yes_bid_dollars"))
    if yb is None:
        yb = _coerce_float(m.get("yes_bid"))
    nb = _coerce_float(m.get("no_bid_dollars"))
    if nb is None:
        nb = _coerce_float(m.get("no_bid"))
    last = _coerce_float(m.get("last_price_dollars"))
    if last is None:
        last = _coerce_float(m.get("last_price"))
    if yb is not None and yb > 0 and nb is not None and nb > 0:
        return ((yb + (1 - nb)) / 2.0, "quote_2sided")
    if yb is not None and yb > 0:
        return (yb, "yes_bid")
    if nb is not None and nb > 0:
        return (1 - nb, "inv_no_bid")
    if last is not None and last > 0:
        return (last, "last_price")
    return (None, "none")


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
    """Summarize a market for the forecaster brief.

    Adds two fields beyond the original schema (backward-compatible — old
    fields are preserved):
      - ``outcome``: the candidate label this binary market represents
        (e.g. "Janice STFU", "Above 4.510"). Pulled from yes_sub_title or
        expiration_value. The forecaster needs this to match against
        ``event.outcomes``; the generic ``title`` is the same string across
        every market in an event and is not enough on its own.
      - ``price_source``: where ``yes_price`` came from (quote_2sided /
        yes_bid / inv_no_bid / last_price / none). Helps the forecaster
        weight the anchor — a 2-sided live quote is stronger than a stale
        last_price.
    """
    yes_price, price_source = _compute_yes_price(m)
    return {
        "keyword": keyword,
        "ticker": m.get("ticker", ""),
        "event_ticker": m.get("event_ticker", ""),
        "outcome": m.get("yes_sub_title") or m.get("expiration_value") or "",
        "title": m.get("title", ""),
        "yes_price": yes_price,
        "price_source": price_source,
        # Legacy fields kept null-tolerantly for backward compat with anything
        # downstream that may read them.
        "no_price": _coerce_float(m.get("no_ask_dollars")) or _coerce_float(m.get("no_ask")),
        "volume": m.get("volume_fp") or m.get("volume", 0),
        "open_interest": m.get("open_interest_fp") or m.get("open_interest", 0),
        "close_time": m.get("close_time", ""),
        "status": m.get("status", ""),
    }


def _has_signal(m: dict) -> bool:
    """Keep every market Kalshi returns, even untraded ones.

    Previously this filtered out markets without price/volume/OI to avoid
    diluting the brief — but the public ``/markets`` list endpoint never
    populates ``yes_ask``/``last_price``/``volume`` (those live in the
    per-market ``/orderbook`` endpoint), so the filter was rejecting 100%
    of legitimately-active markets and returning 0 items to the forecaster.
    We now return True unconditionally; the forecaster at least gets the
    candidate outcome set Kalshi has listed for the series.
    """
    return True


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
    # Default True: the current event's own markets are the strongest
    # calibration anchor (live market price for the exact question we're
    # forecasting). Set include_self=False to recover the old behaviour.
    include_self = bool(args.get("include_self", True))

    auto_series = _series_from_ticker(event.event_ticker)
    if auto_series and auto_series not in series_tickers:
        series_tickers.insert(0, auto_series)

    if not series_tickers and not keywords:
        return ResearchResult(
            tool="kalshi_related",
            summary="no series_tickers or keywords; skipping",
        )

    # Empty string never matches a real Kalshi event_ticker, so the
    # equality checks below become no-ops when include_self is True.
    exclude_event = "" if include_self else event.event_ticker
    items: list[dict] = []
    notes: list[str] = []

    async with httpx.AsyncClient(timeout=min(timeout_s, 30.0)) as client:
        # STEP 1: Fetch the current event's full market data via /events
        # endpoint. This is the only public endpoint that returns markets
        # with populated bid/ask/last_price fields — the /markets list we
        # use below for related-series discovery returns sparse data.
        # We add these first so dedup-by-ticker keeps the rich version
        # if the series query later returns the same tickers.
        if include_self:
            ev_data = await _fetch_event_detail(client, event.event_ticker)
            if ev_data:
                self_markets = ev_data.get("markets", []) or []
                priced = 0
                for m in self_markets:
                    summ = _summarize_market(m, keyword="event_self")
                    items.append(summ)
                    if summ.get("yes_price") is not None:
                        priced += 1
                notes.append(
                    f"self event {event.event_ticker}: {len(self_markets)} markets, {priced} priced"
                )
            else:
                notes.append(f"self event {event.event_ticker}: detail fetch failed")

        for series in series_tickers[:5]:
            try:
                markets = await _fetch_series_markets(client, series, limit=100)
            except httpx.HTTPError as exc:
                notes.append(f"series {series}: {exc}")
                continue
            # Snapshot tickers already added (from /events fetch or prior
            # series queries) so we don't waste a slot on a duplicate that
            # the bottom-of-function dedup would drop anyway.
            already_seen = {it["ticker"] for it in items}
            kept = 0
            skipped_no_signal = 0
            for m in markets:
                if m.get("event_ticker") == exclude_event:
                    continue
                if m.get("ticker") in already_seen:
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
            already_seen = {it["ticker"] for it in items}
            for kw in keywords[:5]:
                kw_lower = kw.lower()
                kept = 0
                for m in pool:
                    if m.get("event_ticker") == exclude_event:
                        continue
                    if m.get("ticker") in already_seen:
                        continue
                    title = (m.get("title") or "").lower()
                    ticker = (m.get("ticker") or "").lower()
                    if kw_lower not in title and kw_lower not in ticker:
                        continue
                    if not _has_signal(m):
                        continue
                    items.append(_summarize_market(m, keyword=kw))
                    already_seen.add(m.get("ticker"))
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

    self_note = (
        f"includes self={event.event_ticker}" if include_self else f"excluded self={event.event_ticker}"
    )
    n_priced = sum(1 for it in deduped if it.get("yes_price") is not None)
    summary = (
        f"{len(deduped)} unique markets, {n_priced} with yes_price ({self_note}). "
        + " | ".join(notes)
    )
    return ResearchResult(tool="kalshi_related", items=deduped, summary=summary)
