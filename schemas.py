from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EventRequest(BaseModel):
    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str
    rules: str | None = None
    close_time: str
    outcomes: list[str] | None = None


class MarketProbability(BaseModel):
    market: str
    probability: float


class PredictionResponse(BaseModel):
    probabilities: list[MarketProbability]


class ResearchStep(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ResearchPlan(BaseModel):
    reasoning: str
    # Fermi decomposition: 3-7 specific sub-questions whose answers directly
    # change the forecast. Threaded into research briefs, the synthesizer's
    # output, and the forecaster prompts. Optional for backward compat.
    sub_questions: list[str] = Field(default_factory=list)
    research_steps: list[ResearchStep]


class ResearchResult(BaseModel):
    tool: str
    items: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    error: str | None = None
    elapsed_s: float = 0.0


class ForecastDraft(BaseModel):
    model: str
    probabilities: list[MarketProbability]
    rationale: str
    confidence: str = "medium"


def outcome_labels(event: EventRequest) -> list[str]:
    if event.outcomes:
        return list(event.outcomes)
    return [event.market_ticker]


def floor_for_outcomes(n_outcomes: int) -> float:
    """Per-outcome minimum probability, scaled by outcome count.

    A fixed 0.01 floor crowds high-confidence answers out of many-outcome
    events (22 × 0.01 = 22% mass locked away from the correct answer in a
    23-outcome event). This adapts: smaller per-outcome floor when there are
    more outcomes, with a hard minimum to bound catastrophic Brier on rare
    upsets.

    Examples:
        N=2  -> 0.025  (binary stays gentle)
        N=6  -> 0.0083
        N=23 -> 0.002  (hard minimum kicks in)
    """
    return max(0.002, 0.05 / max(n_outcomes, 1))
