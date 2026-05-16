"""Final aggregator: Opus synthesizes the brief + both forecasts + critique
into a final probability distribution.

Explicitly empowered to disagree with both upstream forecasters if the critique
is persuasive. We don't just average — we synthesize.
"""
from __future__ import annotations

import json
import logging
import re

from llm_clients import CLAUDE_EFFORT, OPUS, claude_complete
from schemas import (
    EventRequest,
    ForecastDraft,
    MarketProbability,
    PredictionResponse,
    outcome_labels,
)

log = logging.getLogger(__name__)

SYSTEM = """You are the final-call forecaster for a calibrated prediction agent.

You will receive:
1. The forecasting event.
2. The evidence brief.
3. Two independent forecasts with rationales.
4. A red-team critique of those forecasts.

Your job is to produce the FINAL probability distribution. You can:
- Agree with the consensus when the critique is weak.
- Pull toward the dissenting forecast if its reasoning held up better.
- Move BEYOND both forecasts if the critique exposes a flaw they share.

CALIBRATION DISCIPLINE
- Probabilities must sum to 1.0 and use the EXACT outcome labels.
- Clamp each probability to [0.01, 0.99].
- Prefer the cross-market consensus unless the evidence brief provides specific \
information beyond what the market has priced in.
- If the two upstream forecasts agree closely AND the critique is mild, stay \
near their consensus. If they disagree or the critique is sharp, reason from \
first principles.

OUTPUT
Respond with ONLY valid JSON (no markdown, no preamble):
{
  "probabilities": [{"market": "<exact_outcome>", "probability": <float>}, ...],
  "rationale": "<3-5 sentences explaining the final call and what swung you>"
}"""


def _user_prompt(
    event: EventRequest,
    brief: str,
    drafts: list[ForecastDraft],
    critique: str,
) -> str:
    drafts_block = "\n\n".join(
        f"FORECAST {i+1} ({d.model}, confidence={d.confidence})\n"
        f"  probabilities: "
        + ", ".join(f"{p.market}={p.probability:.3f}" for p in d.probabilities)
        + f"\n  rationale: {d.rationale}"
        for i, d in enumerate(drafts)
    )
    outcomes = outcome_labels(event)
    return (
        f"EVENT: {event.title}\n"
        f"CATEGORY: {event.category}\n"
        f"CLOSE TIME: {event.close_time}\n"
        f"OUTCOMES (use these exact labels): {outcomes}\n\n"
        f"EVIDENCE BRIEF\n{brief}\n\n"
        f"FORECASTS\n{drafts_block}\n\n"
        f"DEVIL'S ADVOCATE CRITIQUE\n{critique}\n\n"
        f"Produce the FINAL forecast as JSON now."
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _safe_final(
    text: str, event: EventRequest, drafts: list[ForecastDraft]
) -> PredictionResponse:
    outcomes = outcome_labels(event)
    try:
        m = _JSON_RE.search(text or "")
        if not m:
            raise ValueError("no JSON")
        data = json.loads(m.group(0))
        probs: dict[str, float] = {}
        for item in data.get("probabilities", []):
            market = str(item.get("market", "")).strip()
            try:
                p = float(item.get("probability", 0))
            except (TypeError, ValueError):
                continue
            if p > 1.0:
                p = p / 100.0
            if market in outcomes:
                probs[market] = max(0.01, min(0.99, p))
        if probs:
            for o in outcomes:
                probs.setdefault(o, 0.01)
            total = sum(probs.values())
            return PredictionResponse(
                probabilities=[
                    MarketProbability(market=o, probability=probs[o] / total)
                    for o in outcomes
                ]
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("aggregator parse failed (%s); falling back to draft mean", exc)

    return _mean_of_drafts(event, drafts)


def _mean_of_drafts(
    event: EventRequest, drafts: list[ForecastDraft]
) -> PredictionResponse:
    outcomes = outcome_labels(event)
    if not drafts:
        p = 1.0 / len(outcomes)
        return PredictionResponse(
            probabilities=[MarketProbability(market=o, probability=p) for o in outcomes]
        )
    sums = {o: 0.0 for o in outcomes}
    for d in drafts:
        for mp in d.probabilities:
            if mp.market in sums:
                sums[mp.market] += mp.probability
    total = sum(sums.values()) or 1.0
    return PredictionResponse(
        probabilities=[
            MarketProbability(market=o, probability=sums[o] / total) for o in outcomes
        ]
    )


async def aggregate(
    event: EventRequest,
    brief: str,
    drafts: list[ForecastDraft],
    critique: str,
) -> PredictionResponse:
    try:
        text = await claude_complete(
            model=OPUS,
            system=SYSTEM,
            user=_user_prompt(event, brief, drafts, critique),
            max_tokens=2000,
            effort=CLAUDE_EFFORT,
        )
        return _safe_final(text, event, drafts)
    except Exception as exc:  # noqa: BLE001
        log.warning("aggregator failed (%s); falling back to draft mean", exc)
        return _mean_of_drafts(event, drafts)
