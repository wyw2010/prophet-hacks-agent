"""Parallel forecasters: Opus 4.7 and GPT-5 produce independent forecasts.

Both receive an identical system prompt and the same evidence brief, so any
divergence reflects genuine model disagreement rather than prompt drift.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from llm_clients import CLAUDE_EFFORT, GPT, OPUS, claude_complete, gpt_complete, openai_client
from schemas import EventRequest, ForecastDraft, MarketProbability, floor_for_outcomes, outcome_labels

log = logging.getLogger(__name__)

FORECAST_SYSTEM = """You are an elite calibrated forecaster. You will receive:
1. The forecasting event (title, outcomes, category, rules).
2. An evidence brief synthesized from multiple research sources.

Your job is to produce a probability distribution over the event's outcomes.

OUTCOME INTERPRETATION (CRITICAL)
Outcomes are ALWAYS mutually exclusive — exactly one will resolve true, and \
probabilities must sum to 1.0.

For threshold-style outcomes ("Above X%", "Below X", "At least N"), each \
outcome represents a BUCKET, not a cumulative probability. The winning outcome \
is the one whose threshold is the HIGHEST value ≤ the resolved actual.

  Example: outcomes ["Above 3.25%", "Above 3.50%", "Above 3.75%", "Above 4.00%"]
  with actual rate = 3.75% → "Above 3.75%" wins (bucket [3.75%, 4.00%)).

  Probabilities do NOT decrease monotonically. They concentrate on the bucket \
you think the resolved value will land in. A flat decreasing distribution \
(0.30, 0.25, 0.20, 0.15) signals you're treating these as cumulative — wrong. \
A concentrated distribution (0.05, 0.15, 0.60, 0.20) is what bucket forecasts \
look like.

For "Exactly X" outcomes, same logic — pick the bucket you think will resolve.

REASONING WORKFLOW (Fermi-style)
Before assigning probabilities, work through the brief's "Sub-question answers" \
section. For each sub-question, jot down a one-sentence inference: \
"This evidence points toward outcome <X> because <Y>." Only then commit to \
probabilities. This keeps your forecast anchored to specific evidence rather \
than gestalt vibes.

CALIBRATION DISCIPLINE
- Probabilities MUST sum to 1.0 across all outcomes.
- Use the EXACT outcome labels supplied. Do not invent or rename outcomes.
- Anchor on cross-market signals (Kalshi-related, Polymarket-related, sportsbook \
implied) when present — they aggregate many minds and are hard to beat unless \
you have specific information.
- Move away from market anchors only when the evidence brief surfaces information \
the market plausibly hasn't priced.
- Avoid 0.95+ or 0.05- unless the evidence is overwhelming. HOWEVER: if the \
brief explicitly states the event has already resolved, the outcome is \
factually documented in multiple sources, or the contest is mathematically \
decided, the correct probability for the documented outcome is 0.95+ (often \
0.98–0.99). Do not apply uncertainty discounting to past facts.

OUTPUT
Respond with ONLY valid JSON (no markdown, no preamble):
{
  "probabilities": [{"market": "<exact_outcome>", "probability": <float>}, ...],
  "rationale": "<3-5 sentences citing specific evidence and base rates>",
  "confidence": "low" | "medium" | "high"
}"""


def _user_prompt(event: EventRequest, brief: str) -> str:
    outcomes = outcome_labels(event)
    return (
        f"EVENT: {event.title}\n"
        f"CATEGORY: {event.category}\n"
        f"CLOSE TIME: {event.close_time}\n"
        f"OUTCOMES (use these exact labels): {outcomes}\n"
        f"RULES: {event.rules or '(none)'}\n\n"
        f"EVIDENCE BRIEF\n{brief}\n\n"
        f"Produce your forecast as JSON now."
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_forecast(text: str, event: EventRequest, model: str) -> ForecastDraft:
    outcomes = outcome_labels(event)
    floor = floor_for_outcomes(len(outcomes))
    ceiling = 1.0 - floor
    m = _JSON_RE.search(text or "")
    if not m:
        raise ValueError("no JSON in forecast output")
    data = json.loads(m.group(0))

    raw = data.get("probabilities", [])
    probs: dict[str, float] = {}
    for item in raw:
        market = str(item.get("market", "")).strip()
        try:
            p = float(item.get("probability", 0))
        except (TypeError, ValueError):
            continue
        if p > 1.0:
            p = p / 100.0
        if market in outcomes:
            probs[market] = max(floor, min(ceiling, p))

    # Outcomes the model omitted are presumed near-zero (use floor, not uniform).
    for o in outcomes:
        probs.setdefault(o, floor)

    total = sum(probs.values()) or 1.0
    normalized = [
        MarketProbability(market=o, probability=probs[o] / total) for o in outcomes
    ]

    return ForecastDraft(
        model=model,
        probabilities=normalized,
        rationale=data.get("rationale", ""),
        confidence=data.get("confidence", "medium"),
    )


def _uniform(event: EventRequest, model: str, reason: str) -> ForecastDraft:
    outcomes = outcome_labels(event)
    p = 1.0 / len(outcomes)
    return ForecastDraft(
        model=model,
        probabilities=[MarketProbability(market=o, probability=p) for o in outcomes],
        rationale=f"uniform fallback: {reason}",
        confidence="low",
    )


async def _opus_forecast(event: EventRequest, brief: str) -> ForecastDraft:
    try:
        text = await claude_complete(
            model=OPUS,
            system=FORECAST_SYSTEM,
            user=_user_prompt(event, brief),
            max_tokens=2000,
            effort=CLAUDE_EFFORT,
        )
        return _parse_forecast(text, event, OPUS)
    except Exception as exc:  # noqa: BLE001
        log.warning("opus forecast failed: %s", exc)
        return _uniform(event, OPUS, str(exc))


async def _gpt_forecast(event: EventRequest, brief: str) -> ForecastDraft:
    if openai_client() is None:
        return _uniform(event, GPT, "OPENAI_API_KEY not set")
    try:
        text = await gpt_complete(
            model=GPT,
            system=FORECAST_SYSTEM,
            user=_user_prompt(event, brief),
            max_tokens=16000,
            reasoning_effort="high",
        )
        return _parse_forecast(text, event, GPT)
    except Exception as exc:  # noqa: BLE001
        log.warning("gpt forecast failed: %s", exc)
        return _uniform(event, GPT, str(exc))


async def forecast_both(event: EventRequest, brief: str) -> list[ForecastDraft]:
    return await asyncio.gather(_opus_forecast(event, brief), _gpt_forecast(event, brief))
