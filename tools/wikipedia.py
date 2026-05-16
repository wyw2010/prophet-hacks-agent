"""Wikipedia lookups via the MediaWiki action API.

We previously used the REST summary endpoint (``/api/rest_v1/page/summary/...``)
but it now serves 403 to many cloud-IP User-Agents even when the UA is properly
formed. The classic action API has looser policies and returns the same intro
extract via ``prop=extracts&exintro=true&explaintext=true``.
"""
from __future__ import annotations

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register


WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_UA = (
    "prophet-hacks-agent/0.1 "
    "(https://github.com/wyw2010/prophet-hacks-agent)"
)


@register(
    name="wikipedia",
    description=(
        "Wikipedia article intro extracts via the MediaWiki action API. "
        "Use for background on entities — athletes, companies, countries, "
        "scientific terms, awards ceremonies. Cheap and rate-limit-friendly."
    ),
    args_schema='{"titles": ["Cleveland Cavaliers", "2026 NBA playoffs"]}',
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    titles: list[str] = [t for t in (args.get("titles") or []) if t][:8]
    if not titles:
        return ResearchResult(tool="wikipedia", summary="no titles supplied; skipping")

    items: list[dict] = []
    async with httpx.AsyncClient(
        timeout=min(timeout_s, 20.0),
        headers={"User-Agent": WIKI_UA, "Accept": "application/json"},
    ) as client:
        try:
            r = await client.get(
                WIKI_API,
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "exintro": "true",
                    "explaintext": "true",
                    "redirects": "1",
                    "titles": "|".join(titles),
                },
            )
            r.raise_for_status()
            payload = r.json()
        except httpx.HTTPError as exc:
            return ResearchResult(tool="wikipedia", error=f"http error: {exc}")

    pages = payload.get("query", {}).get("pages", {}) or {}
    found_titles: set[str] = set()
    for page in pages.values():
        page_title = page.get("title", "")
        found_titles.add(page_title)
        extract = page.get("extract", "")
        missing = "missing" in page  # page doesn't exist on Wikipedia
        items.append(
            {
                "title": page_title,
                "extract": extract[:3000] if extract else "",
                "missing": missing,
                "url": f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
            }
        )

    summary = (
        f"Fetched {len(items)} Wikipedia entries "
        f"({sum(1 for i in items if i['missing'])} missing pages)."
    )
    return ResearchResult(tool="wikipedia", items=items, summary=summary)
