"""Anthropic native web search via the `web_search_20250305` tool.

Lets Claude run multiple search round-trips inside one call. We give it the
event context and a research goal; it returns a synthesized findings block.
"""
from __future__ import annotations

import anthropic

from llm_clients import SONNET, anthropic_client
from schemas import EventRequest, ResearchResult
from tools.base import register


SYSTEM = """You are a research assistant for a forecasting agent. Use the web \
search tool to gather current, factual information relevant to the forecasting \
question. Cite sources inline by URL. Be terse — bullet points only."""


@register(
    name="claude_news",
    description=(
        "Anthropic-hosted web search. Best for current events and breaking news. "
        "The model runs multiple searches autonomously; you give it a brief."
    ),
    args_schema='{"brief": "natural-language research goal", "max_searches": 3}',
    requires_env=["ANTHROPIC_API_KEY"],
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    brief = args.get("brief") or f"Find evidence relevant to: {event.title}"
    max_searches = int(args.get("max_searches", 3))

    user = (
        f"Forecasting question: {event.title}\n"
        f"Category: {event.category}\n"
        f"Close time: {event.close_time}\n\n"
        f"Research goal: {brief}\n\n"
        "Return 5-10 bulleted findings, each with a source URL."
    )

    try:
        response = await anthropic_client().messages.create(
            model=SONNET,
            max_tokens=2048,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": max_searches,
                }
            ],
        )
    except anthropic.APIError as exc:
        return ResearchResult(tool="claude_news", error=f"anthropic api: {exc}")

    text_blocks = [
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ]
    summary = "\n".join(text_blocks).strip()

    citations: list[dict] = []
    for block in response.content:
        for c in getattr(block, "citations", None) or []:
            citations.append(
                {
                    "url": getattr(c, "url", ""),
                    "title": getattr(c, "title", ""),
                }
            )

    return ResearchResult(tool="claude_news", items=citations, summary=summary)
