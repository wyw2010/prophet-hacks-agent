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

Your job has TWO parts:

PART 1: FERMI DECOMPOSITION (Tetlock-style)
Decompose the forecasting question into 3-7 specific sub-questions whose \
answers would directly change the probability estimate. These are the \
empirical questions a superforecaster would research before committing to a \
number. They must be SPECIFIC and ANSWERABLE from public sources.

Good sub-questions (for "Will the Fed cut rates in September 2026?"):
  - What is the current upper bound of the federal funds rate?
  - What probability does the OIS/FedWatch curve imply for a September cut?
  - What did the most recent dot-plot project for year-end 2026?
  - What is the most recent CPI/PCE reading vs. the Fed's 2% target?
  - What is the new chair Warsh's prior policy stance?
  - Are there active dissenters favoring cuts or hikes?

Bad sub-questions (too vague or unanswerable):
  - Will the Fed do the right thing?
  - What does the future hold for monetary policy?

PART 2: RESEARCH PLAN
Pick tools that will help answer the sub-questions. Feed the sub-questions \
into tool args where it makes sense — e.g., put them in `claude_news.brief` \
or as `mengye_search.queries` so the research is steered.

GUIDELINES
- Always consider news search (mengye_search and/or claude_news) unless the \
event is purely numeric/statistical.
- Always consider kalshi_related — cross-market signals are cheap and informative. \
Pick keywords broader than the event itself.
- DO NOT invoke any tool to look up the event's exact ticker — that's circular.
- For each tool you pick, supply realistic args following its args schema.

TOOL-PICKING HEURISTICS (when to reach for each)
The full per-tool descriptions and arg schemas are listed below. Quick mental map:
- Economics / monetary policy / inflation / employment → fred (FRED series IDs)
- Public-company stocks, IPOs, earnings beats/misses, stock price thresholds \
→ earnings_data (pass tickers like AAPL, NVDA)
- Cryptocurrency prices, on-chain → crypto (pass CoinGecko ids: bitcoin, ethereum)
- US federal legislation (bills, votes) → congress_bills
- Court cases, rulings, convictions, sentencing, scheduled hearings, SCOTUS \
→ court_docket (free-text query + type='o' for opinions or 'd' for active dockets)
- Sports games (h2h) or championship futures → sports_odds (use sport keys ending \
in _championship_winner for futures)
- Background on entities, awards, athletes → wikipedia
- ANY question with non-trivial arithmetic (de-vigging odds, base-rate math, \
normalizing distributions across many outcomes, monte carlo) → code_execution. \
Cheap insurance against the forecasters miscounting or hand-waving math.

Use the catalog below to confirm exact arg shapes before emitting your plan.

OUTPUT
Respond with ONLY valid JSON (no markdown, no preamble):
{
  "reasoning": "2-3 sentences on what info matters most and which sources you chose",
  "sub_questions": [
    "Specific empirical sub-question 1?",
    "Specific empirical sub-question 2?",
    ...
  ],
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
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.MULTILINE)


def _extract_json(text: str) -> dict:
    """Extract + parse JSON from the model's output.

    Robust to: markdown code fences, leading/trailing prose, and the model
    accidentally including the trailing brace from a comment block. Tries
    fences first (cleanest), then a brace-balanced scan, then a greedy
    fallback regex.
    """
    text = text.strip()

    # Strip markdown code fence if present
    fence = _CODE_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    # Brace-balanced scan starting at the first {. This handles cases where
    # there's trailing prose after the JSON object.
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Last resort: greedy match
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON object in planner output: {text[:200]!r}")
    return json.loads(m.group(0))


async def plan(event: EventRequest) -> ResearchPlan:
    catalog = TOOL_CATALOG()
    # Bumped 2500 -> 6000. With 12 tools, Fermi sub-questions, and verbose
    # reasoning fields, the planner regularly produced 1500-2200 token outputs.
    # 2500 was tight enough that any verbosity truncated the JSON mid-args.
    text = await claude_complete(
        model=OPUS,
        system=SYSTEM,
        user=_user_prompt(event, catalog),
        max_tokens=6000,
    )
    try:
        data = _extract_json(text)
        steps = [ResearchStep(**s) for s in data.get("research_steps", [])]
        sub_qs = [str(q).strip() for q in data.get("sub_questions", []) if str(q).strip()]
        return ResearchPlan(
            reasoning=data.get("reasoning", ""),
            sub_questions=sub_qs,
            research_steps=steps,
        )
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        log.warning("planner returned invalid JSON (%s); falling back to defaults", exc)
        return _fallback_plan(event)


def _fallback_plan(event: EventRequest) -> ResearchPlan:
    """If the planner fails, default to a broad sweep so we still produce a forecast.

    This fallback is dumb on purpose — no domain branching, just a wide net of
    tools that are useful regardless of category. Includes code_execution and
    a category-conditional FRED/earnings call so we don't fully whiff when the
    planner truncates on a math-heavy or stock-related question.
    """
    keywords = [w for w in re.findall(r"[A-Za-z]{4,}", event.title)][:4] or [event.title]
    category = (event.category or "").lower()

    steps: list[ResearchStep] = [
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
            tool="code_execution",
            args={
                "task": (
                    f"Compute base rates and any relevant arithmetic for: "
                    f"{event.title}. Outcomes: {event.outcomes}."
                ),
                "max_uses": 2,
            },
        ),
    ]

    # Category-conditional adds
    if "econom" in category or "finance" in category or "rate" in category:
        steps.append(
            ResearchStep(
                tool="fred",
                args={
                    "series_ids": ["DFEDTARU", "DFF", "CPIAUCSL", "UNRATE", "DGS10"],
                    "lookback_days": 365,
                },
            )
        )

    return ResearchPlan(
        reasoning="planner fallback: broad sweep across always-useful tools + category conditionals",
        sub_questions=[
            f"What is the most recent factual status of: {event.title}?",
            "What do prediction markets currently price for this event?",
            "What recent news affects the probability of each outcome?",
            "What is the historical base rate for similar events?",
        ],
        research_steps=steps,
    )
