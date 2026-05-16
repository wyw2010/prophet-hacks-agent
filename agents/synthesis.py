"""Synthesis: Sonnet condenses raw research into a structured evidence brief."""
from __future__ import annotations

import json
import logging
from typing import Iterable

from llm_clients import SONNET, claude_complete
from schemas import EventRequest, ResearchResult

log = logging.getLogger(__name__)

SYSTEM = """You are the research synthesizer for a forecasting agent.

You receive raw research output from multiple tools and produce a structured \
evidence brief that the forecaster can act on.

OUTCOME INTERPRETATION
Outcomes are mutually exclusive — exactly one will resolve true. For \
threshold-style outcomes ("Above X%", "Below X", "At least N"), each is a \
BUCKET: the winning outcome is the one whose threshold is the highest value \
≤ the resolved actual. When describing directional leans, anchor on the BUCKET \
each outcome represents, not cumulative probabilities.

OUTPUT FORMAT (markdown, no JSON wrapper):

# Event
<one-line restatement>

# Outcomes to forecast
<list>

# Key facts (high-confidence, factual)
1. [source] fact
2. ...

# Cross-market signals
- Kalshi related: brief synthesis of related-market prices
- Polymarket related: brief synthesis
- Sportsbook implied (if present): brief synthesis

# Analyst opinions and speculation
- ...

# Directional lean per outcome
- <outcome A>: <supporting evidence count vs. opposing>
- <outcome B>: <supporting evidence count vs. opposing>

# Gaps / unknowns
- ...

# Calibration anchors
- base rates from related markets or odds
- recent precedents for similar events

Keep total length under 800 words. Be terse, factual, and source-tag every claim."""


def _format_research(results: Iterable[ResearchResult]) -> str:
    chunks: list[str] = []
    for r in results:
        if r.error:
            chunks.append(f"## TOOL: {r.tool} (FAILED)\nerror: {r.error}\n")
            continue
        chunks.append(f"## TOOL: {r.tool}")
        if r.summary:
            chunks.append(f"summary: {r.summary}")
        if r.items:
            items_json = json.dumps(r.items[:30], indent=2, default=str)
            if len(items_json) > 8000:
                items_json = items_json[:8000] + "\n... (truncated)"
            chunks.append(f"items:\n{items_json}")
        chunks.append("")
    return "\n".join(chunks)


async def synthesize(event: EventRequest, results: list[ResearchResult]) -> str:
    user = (
        f"EVENT: {event.title}\n"
        f"CATEGORY: {event.category}\n"
        f"OUTCOMES: {event.outcomes or [event.market_ticker]}\n"
        f"CLOSE TIME: {event.close_time}\n"
        f"DESCRIPTION: {event.description or '(none)'}\n"
        f"RULES: {event.rules or '(none)'}\n\n"
        f"RAW RESEARCH\n{_format_research(results)}\n\n"
        f"Produce the evidence brief now."
    )
    return await claude_complete(
        model=SONNET,
        system=SYSTEM,
        user=user,
        max_tokens=3000,
    )
