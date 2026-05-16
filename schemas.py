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
