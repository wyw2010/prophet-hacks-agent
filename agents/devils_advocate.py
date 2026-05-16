"""Devil's Advocate: Opus red-teams the two forecasts.

Surfaces missed scenarios, overweighted evidence, and over/under-confidence
that could push the final aggregator off the consensus.
"""
from __future__ import annotations

from llm_clients import CLAUDE_EFFORT, OPUS, claude_complete
from schemas import EventRequest, ForecastDraft

SYSTEM = """You are a red-team forecaster. You will receive:
1. A forecasting event and evidence brief.
2. Two independent forecasts (from different models) including their rationales.

Your job is to find what they missed:
- Scenarios neither considered (especially low-probability tail outcomes)
- Evidence that should weigh more or less than they gave it
- Overconfidence (probabilities too extreme given the actual evidence)
- Underconfidence (defaulting toward 50/50 when evidence is clearly directional)
- Reference-class errors (the wrong historical comparison)

Be specific. Quote evidence from the brief. Do not produce a probability yourself \
— your output is purely diagnostic for the aggregator.

OUTPUT
3-6 numbered critiques, each ≤2 sentences. Plain text, no markdown headers."""


def _user_prompt(event: EventRequest, brief: str, drafts: list[ForecastDraft]) -> str:
    drafts_block = "\n\n".join(
        f"FORECAST {i+1} ({d.model}, confidence={d.confidence})\n"
        f"  probabilities: "
        + ", ".join(f"{p.market}={p.probability:.3f}" for p in d.probabilities)
        + f"\n  rationale: {d.rationale}"
        for i, d in enumerate(drafts)
    )
    return (
        f"EVENT: {event.title}\n"
        f"CATEGORY: {event.category}\n"
        f"OUTCOMES: {event.outcomes or [event.market_ticker]}\n"
        f"CLOSE TIME: {event.close_time}\n\n"
        f"EVIDENCE BRIEF\n{brief}\n\n"
        f"FORECASTS TO CRITIQUE\n{drafts_block}\n\n"
        f"Produce your critique now."
    )


async def critique(
    event: EventRequest, brief: str, drafts: list[ForecastDraft]
) -> str:
    return await claude_complete(
        model=OPUS,
        system=SYSTEM,
        user=_user_prompt(event, brief, drafts),
        max_tokens=1500,
        effort=CLAUDE_EFFORT,
    )
