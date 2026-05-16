"""CoinGecko price lookups (no key required for the free public API)."""
from __future__ import annotations

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register


CG_BASE = "https://api.coingecko.com/api/v3"


@register(
    name="crypto",
    description=(
        "CoinGecko spot price, 24h change, and 30d historical data for crypto assets. "
        "Args take CoinGecko ids (e.g. ``bitcoin``, ``ethereum``, ``solana``)."
    ),
    args_schema='{"asset_ids": ["bitcoin", "ethereum"], "history_days": 30}',
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    asset_ids: list[str] = args.get("asset_ids") or []
    history_days = int(args.get("history_days", 30))
    if not asset_ids:
        return ResearchResult(tool="crypto", summary="no asset_ids supplied; skipping")

    items: list[dict] = []
    async with httpx.AsyncClient(timeout=min(timeout_s, 30.0)) as client:
        try:
            r = await client.get(
                f"{CG_BASE}/simple/price",
                params={
                    "ids": ",".join(asset_ids[:10]),
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                    "include_market_cap": "true",
                },
            )
            r.raise_for_status()
            spot = r.json()
        except httpx.HTTPError as exc:
            return ResearchResult(tool="crypto", error=f"http error (spot): {exc}")

        for aid in asset_ids[:5]:
            entry: dict = {"asset_id": aid, "spot": spot.get(aid, {})}
            try:
                hr = await client.get(
                    f"{CG_BASE}/coins/{aid}/market_chart",
                    params={"vs_currency": "usd", "days": history_days, "interval": "daily"},
                )
                hr.raise_for_status()
                hist = hr.json()
                prices = hist.get("prices", [])
                if prices:
                    first, last = prices[0][1], prices[-1][1]
                    entry["history"] = {
                        "days": history_days,
                        "first": first,
                        "last": last,
                        "pct_change": round((last - first) / first * 100, 2) if first else None,
                    }
            except httpx.HTTPError:
                pass
            items.append(entry)

    summary = f"Fetched spot + {history_days}d history for {len(items)} crypto asset(s)."
    return ResearchResult(tool="crypto", items=items, summary=summary)
