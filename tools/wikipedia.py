"""Wikipedia REST summary lookups."""
from __future__ import annotations

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register


WIKI_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"


@register(
    name="wikipedia",
    description=(
        "Wikipedia article summaries. Use for background on entities — athletes, "
        "companies, countries, scientific terms. Cheap and rate-limit-friendly."
    ),
    args_schema='{"titles": ["Cleveland Cavaliers", "2026 NBA playoffs"]}',
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    titles: list[str] = args.get("titles") or []
    if not titles:
        return ResearchResult(tool="wikipedia", summary="no titles supplied; skipping")

    items: list[dict] = []
    async with httpx.AsyncClient(
        timeout=min(timeout_s, 15.0),
        headers={"User-Agent": "prophet-hacks-agent/0.1 (contact: github.com/wyw2010)"},
    ) as client:
        for title in titles[:8]:
            try:
                r = await client.get(f"{WIKI_BASE}/{title.replace(' ', '_')}")
                if r.status_code == 404:
                    items.append({"title": title, "error": "not found"})
                    continue
                r.raise_for_status()
                payload = r.json()
                items.append(
                    {
                        "title": payload.get("title", title),
                        "extract": payload.get("extract", ""),
                        "url": payload.get("content_urls", {})
                        .get("desktop", {})
                        .get("page", ""),
                    }
                )
            except httpx.HTTPError as exc:
                items.append({"title": title, "error": str(exc)})

    summary = f"Fetched {len(items)} Wikipedia summaries."
    return ResearchResult(tool="wikipedia", items=items, summary=summary)
