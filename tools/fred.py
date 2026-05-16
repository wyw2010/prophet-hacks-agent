"""FRED economic data lookups.

Planner specifies series IDs (e.g. ``DFF``, ``CPIAUCSL``, ``UNRATE``) and we
return the most recent observations.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register


@register(
    name="fred",
    description=(
        "Federal Reserve Economic Data. Useful for: rates (DFF, DGS10), inflation "
        "(CPIAUCSL, T10YIE), employment (UNRATE, PAYEMS), GDP (GDPC1). "
        "Returns last ~12 observations per series."
    ),
    args_schema='{"series_ids": ["DFF", "CPIAUCSL"], "lookback_days": 365}',
    requires_env=["FRED_API_KEY"],
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    key = os.environ["FRED_API_KEY"]
    series_ids: list[str] = args.get("series_ids") or []
    lookback_days = int(args.get("lookback_days", 365))
    if not series_ids:
        return ResearchResult(tool="fred", summary="no series_ids supplied; skipping")

    start = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    items: list[dict] = []
    async with httpx.AsyncClient(timeout=min(timeout_s, 30.0)) as client:
        for sid in series_ids[:6]:
            try:
                r = await client.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={
                        "series_id": sid,
                        "api_key": key,
                        "file_type": "json",
                        "observation_start": start,
                        "sort_order": "desc",
                        "limit": 12,
                    },
                )
                r.raise_for_status()
                payload = r.json()
            except httpx.HTTPError as exc:
                items.append({"series_id": sid, "error": str(exc)})
                continue

            obs = [
                {"date": o["date"], "value": o["value"]}
                for o in payload.get("observations", [])
                if o.get("value") not in (".", "")
            ]
            items.append({"series_id": sid, "observations": obs})

    summary = f"Fetched {len(items)} FRED series (lookback={lookback_days}d)."
    return ResearchResult(tool="fred", items=items, summary=summary)
