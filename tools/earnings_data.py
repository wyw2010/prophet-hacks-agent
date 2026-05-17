"""Financial Modeling Prep API for stock quotes, earnings, and fundamentals.

Free tier requires an API key (signup: https://site.financialmodelingprep.com/
developer/docs). 250 requests/day on the free plan — plenty for our usage.

We chose FMP over Yahoo Finance because Yahoo's public endpoints aggressively
rate-limit unauthenticated clients (429 on shared IPs). FMP is reliable and
returns clean JSON.

Endpoints used:
  /api/v3/quote/{ticker}              — current price, P/E, market cap
  /api/v3/historical-price-full/...   — daily history
  /api/v3/earnings-surprises/{ticker} — recent actual vs estimate
  /api/v3/analyst-stock-recommendations/{ticker} — analyst ratings
"""
from __future__ import annotations

import os

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register


FMP_BASE = "https://financialmodelingprep.com/api/v3"


@register(
    name="earnings_data",
    description=(
        "Stock price, market cap, P/E ratios, recent earnings beats/misses, "
        "and analyst recommendations via Financial Modeling Prep. Use for any "
        "event about public companies — earnings questions, stock-price "
        "thresholds, IPO performance. Pass ticker symbols."
    ),
    args_schema='{"tickers": ["AAPL", "NVDA", "TSLA"], "include_history": true, "history_days": 30}',
    requires_env=["FMP_API_KEY"],
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    key = os.environ["FMP_API_KEY"]
    tickers: list[str] = [
        t.strip().upper() for t in (args.get("tickers") or []) if t and str(t).strip()
    ][:6]
    include_history = bool(args.get("include_history", True))
    history_days = max(1, min(int(args.get("history_days", 30)), 365))

    if not tickers:
        return ResearchResult(tool="earnings_data", summary="no tickers supplied")

    items: list[dict] = []
    async with httpx.AsyncClient(timeout=min(timeout_s, 30.0)) as client:
        for ticker in tickers:
            entry: dict = {"ticker": ticker}

            # 1. Current quote (price, P/E, market cap, etc.)
            try:
                r = await client.get(
                    f"{FMP_BASE}/quote/{ticker}",
                    params={"apikey": key},
                )
                r.raise_for_status()
                payload = r.json()
                if isinstance(payload, list) and payload:
                    q = payload[0]
                    entry["price"] = q.get("price")
                    entry["change_pct"] = q.get("changesPercentage")
                    entry["market_cap"] = q.get("marketCap")
                    entry["pe_trailing"] = q.get("pe")
                    entry["volume"] = q.get("volume")
                    entry["year_low"] = q.get("yearLow")
                    entry["year_high"] = q.get("yearHigh")
            except httpx.HTTPError as exc:
                entry["quote_error"] = str(exc)

            # 2. Recent earnings surprises (actual vs estimate, last few quarters)
            try:
                r = await client.get(
                    f"{FMP_BASE}/earnings-surprises/{ticker}",
                    params={"apikey": key},
                )
                r.raise_for_status()
                payload = r.json()
                if isinstance(payload, list):
                    entry["recent_earnings"] = [
                        {
                            "date": e.get("date"),
                            "actual_eps": e.get("actualEarningResult"),
                            "estimated_eps": e.get("estimatedEarning"),
                        }
                        for e in payload[:4]
                    ]
            except httpx.HTTPError as exc:
                entry["earnings_error"] = str(exc)

            # 3. Recent price history
            if include_history:
                try:
                    r = await client.get(
                        f"{FMP_BASE}/historical-price-full/{ticker}",
                        params={"apikey": key, "timeseries": history_days},
                    )
                    r.raise_for_status()
                    payload = r.json()
                    historical = payload.get("historical") or []
                    closes = [h.get("close") for h in historical if h.get("close") is not None]
                    if len(closes) >= 2:
                        first, last = closes[-1], closes[0]  # FMP returns newest first
                        entry["price_history"] = {
                            "days": len(closes),
                            "first_close": round(first, 2),
                            "last_close": round(last, 2),
                            "pct_change": round((last - first) / first * 100, 2) if first else None,
                        }
                except httpx.HTTPError as exc:
                    entry["history_error"] = str(exc)

            items.append(entry)

    summary = f"Fetched FMP data for {len(items)} ticker(s)."
    return ResearchResult(tool="earnings_data", items=items, summary=summary)
