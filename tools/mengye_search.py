"""Custom semantic news search server (port-forwarded by the team).

POST {url} with {"query", "start_date", "end_date", "top_k"} → returns articles.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register


@register(
    name="mengye_search",
    description=(
        "Semantic search over a curated news corpus. Best for finding articles "
        "relevant to a specific event by topic + recency."
    ),
    args_schema='{"queries": ["query 1", "query 2"], "lookback_days": 30, "top_k": 10}',
    requires_env=["MENGYE_SEARCH_URL"],
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    url = os.environ["MENGYE_SEARCH_URL"]
    queries = args.get("queries") or [event.title]
    lookback_days = int(args.get("lookback_days", 30))
    top_k = int(args.get("top_k", 10))

    end = datetime.utcnow().date()
    start = end - timedelta(days=lookback_days)

    items: list[dict] = []
    async with httpx.AsyncClient(timeout=min(timeout_s, 60.0)) as client:
        for q in queries[:5]:  # cap fanout
            try:
                r = await client.post(
                    url,
                    json={
                        "query": q,
                        "start_date": start.isoformat(),
                        "end_date": end.isoformat(),
                        "top_k": top_k,
                    },
                )
                r.raise_for_status()
                payload = r.json()
                results = payload.get("results") if isinstance(payload, dict) else payload
                for art in (results or [])[:top_k]:
                    items.append(
                        {
                            "query": q,
                            "title": art.get("title", ""),
                            "date": art.get("date", ""),
                            "url": art.get("url", ""),
                            "snippet": (art.get("content") or "")[:1000],
                        }
                    )
            except httpx.HTTPError as exc:
                return ResearchResult(
                    tool="mengye_search",
                    items=items,
                    error=f"http error on query {q!r}: {exc}",
                )

    seen, deduped = set(), []
    for it in items:
        key = it["title"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(it)

    summary = f"Fetched {len(deduped)} unique articles across {len(queries)} queries " \
              f"(lookback={lookback_days}d)."
    return ResearchResult(tool="mengye_search", items=deduped, summary=summary)
