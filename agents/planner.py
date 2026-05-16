"""Planner: Opus reads the event + tool catalog, returns a research plan."""
from __future__ import annotations

import json
import logging
import re

from llm_clients import OPUS, claude_complete
from schemas import EventRequest, ResearchPlan, ResearchStep
from tools.base import TOOL_CATALOG

log = logging.getLogger(__name__)

SYSTEM = """You are the research planner for a forecasting agent.

You will be given:
1. A forecasting event with outcomes, category, close_time, and description.
2. A catalog of research tools you can invoke in parallel.

Your job: decide which tools to invoke and with what arguments to gather the \
evidence needed to forecast this event accurately. Be selective — only invoke \
tools that will plausibly contribute. Skip tools whose category doesn't fit.

GUIDELINES
- Always consider news search (mengye_search and/or claude_news) unless the \
event is purely numeric/statistical.
- Always consider kalshi_related and polymarket_related — cross-market signals \
are cheap and informative. Pick keywords that are broader than the event itself \
(e.g. for "Will Cleveland beat Detroit Game 6", use keywords like "cleveland", \
"nba playoffs", NOT the full title).
- DO NOT invoke any tool to look up the event's exact ticker — that's circular.
- For each tool you pick, supply realistic args following its args schema.

OUTPUT
Respond with ONLY valid JSON (no markdown, no preamble):
{
  "reasoning": "2-3 sentences on what info matters most and which sources you chose",
  "research_steps": [
    {"tool": "<tool_name>", "args": {...}},
    ...
  ]
}"""


def _user_prompt(event: EventRequest, catalog: str) -> str:
    return (
        f"EVENT\n"
        f"  event_ticker: {event.event_ticker}\n"
        f"  market_ticker: {event.market_ticker}\n"
        f"  title: {event.title}\n"
        f"  category: {event.category}\n"
        f"  close_time: {event.close_time}\n"
        f"  outcomes: {event.outcomes}\n"
        f"  description: {event.description or '(none)'}\n"
        f"  rules: {event.rules or '(none)'}\n\n"
        f"AVAILABLE TOOLS\n{catalog}\n\n"
        f"Produce the research plan now."
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON object in planner output: {text[:200]!r}")
    return json.loads(m.group(0))


async def plan(event: EventRequest) -> ResearchPlan:
    catalog = TOOL_CATALOG()
    text = await claude_complete(
        model=OPUS,
        system=SYSTEM,
        user=_user_prompt(event, catalog),
        max_tokens=1500,
    )
    try:
        data = _extract_json(text)
        steps = [ResearchStep(**s) for s in data.get("research_steps", [])]
        return ResearchPlan(reasoning=data.get("reasoning", ""), research_steps=steps)
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        log.warning("planner returned invalid JSON (%s); falling back to defaults", exc)
        return _fallback_plan(event)


def _fallback_plan(event: EventRequest) -> ResearchPlan:
    """If the planner fails, default to a broad sweep so we still produce a forecast."""
    keywords = [w for w in re.findall(r"[A-Za-z]{4,}", event.title)][:4] or [event.title]
    return ResearchPlan(
        reasoning="planner fallback: broad sweep across all available tools",
        research_steps=[
            ResearchStep(
                tool="mengye_search",
                args={"queries": [event.title], "lookback_days": 60, "top_k": 10},
            ),
            ResearchStep(
                tool="claude_news",
                args={"brief": f"Find evidence relevant to: {event.title}", "max_searches": 3},
            ),
            ResearchStep(
                tool="kalshi_related",
                args={"keywords": keywords, "limit_per_keyword": 5},
            ),
            ResearchStep(
                tool="polymarket_related",
                args={"keywords": keywords, "limit_per_keyword": 5},
            ),
        ],
    )
