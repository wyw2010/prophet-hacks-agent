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
from schemas import EventRequest, ForecastDraft, MarketProbability, outcome_labels

log = logging.getLogger(__name__)

FORECAST_SYSTEM = """You are an elite calibrated forecaster. You will receive:
1. The forecasting event (title, outcomes, category, rules).
2. An evidence brief synthesized from multiple research sources.

Your job is to produce a probability distribution over the event's outcomes.

CALIBRATION DISCIPLINE
- Probabilities MUST sum to 1.0 across all outcomes.
- Use the EXACT outcome labels supplied. Do not invent or rename outcomes.
- Anchor on cross-market signals (Kalshi-related, Polymarket-related, sportsbook \
implied) when present — they aggregate many minds and are hard to beat unless \
you have specific information.
- Move away from market anchors only when the evidence brief surfaces information \
the market plausibly hasn't priced.
- Avoid 0.95+ or 0.05- unless the evidence is overwhelming.
- Clamp each probability to [0.01, 0.99] before normalizing.

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
            probs[market] = max(0.01, min(0.99, p))

    for o in outcomes:
        probs.setdefault(o, 1.0 / len(outcomes))

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
