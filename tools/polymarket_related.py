"""Polymarket related-markets lookup via Gamma API.

Polymarket's gamma API doesn't have a strong server-side text search, so we
page through active markets and filter client-side. Cross-market signal from
a second prediction-market venue with a different participant base than Kalshi.
"""
from __future__ import annotations

import json
import logging

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register

log = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"


async def _fetch_pool(
    client: httpx.AsyncClient, max_markets: int = 1000
) -> list[dict]:
    """Pull active markets sorted by 24h volume descending.

    Polymarket's default sort is by creation date which surfaces low-volume
    new listings first — exactly the wrong order. Sorting by volume24hr ensures
    we see the high-signal markets in our first few pages. Bumped depth from
    500 -> 1000 so we have a better chance of finding the relevant market.
    """
    out: list[dict] = []
    offset = 0
    page_size = 200
    while len(out) < max_markets:
        r = await client.get(
            f"{GAMMA_BASE}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": page_size,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false",
            },
        )
        r.raise_for_status()
        payload = r.json()
        page = payload if isinstance(payload, list) else payload.get("data", [])
        if not page:
            break
        out.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return out[:max_markets]


def _summarize(m: dict, keyword: str) -> dict:
    raw_prices = m.get("outcomePrices")
    if isinstance(raw_prices, str):
        try:
            raw_prices = json.loads(raw_prices)
        except Exception:  # noqa: BLE001
            raw_prices = None
    outcomes = m.get("outcomes")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except Exception:  # noqa: BLE001
            outcomes = None
    prices_by_outcome: dict[str, float] = {}
    if isinstance(raw_prices, list) and isinstance(outcomes, list):
        for label, price in zip(outcomes, raw_prices):
            try:
                prices_by_outcome[str(label)] = float(price)
            except (TypeError, ValueError):
                continue
    return {
        "keyword": keyword,
        "question": m.get("question", ""),
        "slug": m.get("slug", ""),
        "prices_by_outcome": prices_by_outcome,
        "volume": m.get("volume", 0),
        "end_date": m.get("endDate", ""),
    }


@register(
    name="polymarket_related",
    description=(
        "Find related Polymarket markets by keyword. Pages through active markets "
        "and filters client-side. Different participant base than Kalshi — useful "
        "for triangulating consensus."
    ),
    args_schema='{"keywords": ["bank of japan", "boj rate", "inflation"], "limit_per_keyword": 5}',
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    keywords: list[str] = args.get("keywords") or []
    limit = int(args.get("limit_per_keyword", 5))
    if not keywords:
        return ResearchResult(
            tool="polymarket_related", summary="no keywords supplied; skipping"
        )

    async with httpx.AsyncClient(timeout=min(timeout_s, 30.0)) as client:
        try:
            pool = await _fetch_pool(client, max_markets=500)
        except httpx.HTTPError as exc:
            return ResearchResult(
                tool="polymarket_related", error=f"http error: {exc}"
            )

    items: list[dict] = []
    notes: list[str] = []
    seen_slugs: set[str] = set()
    for kw in keywords[:5]:
        kw_lower = kw.lower()
        kept = 0
        for m in pool:
            slug = m.get("slug") or ""
            if slug in seen_slugs:
                continue
            # Check question, slug, description, group_slug, and tag fields —
            # keywords appear in different places depending on the market type.
            question = (m.get("question") or "").lower()
            description = (m.get("description") or "").lower()
            group_slug = (m.get("groupItemTitle") or m.get("group_slug") or "").lower()
            tags = " ".join(
                (t.get("label") if isinstance(t, dict) else str(t)) or ""
                for t in (m.get("tags") or [])
            ).lower()
            haystack = " | ".join([question, slug.lower(), description, group_slug, tags])
            if kw_lower not in haystack:
                continue
            items.append(_summarize(m, keyword=kw))
            seen_slugs.add(slug)
            kept += 1
            if kept >= limit:
                break
        notes.append(f"keyword {kw!r}: {kept} markets")

    summary = (
        f"Scanned {len(pool)} active Polymarket markets, kept {len(items)} matches. "
        + " | ".join(notes)
    )
    return ResearchResult(tool="polymarket_related", items=items, summary=summary)
