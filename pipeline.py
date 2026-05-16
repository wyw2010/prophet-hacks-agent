"""End-to-end orchestration:

    planner -> parallel research -> synthesis ->
    parallel forecasts (Opus + GPT) -> devil's advocate -> aggregator

Each stage is timeout-isolated. Failure in any single stage degrades gracefully:
research errors are surfaced in the brief; forecaster errors become uniform
fallbacks; aggregator errors fall back to the mean of the drafts.

Every request also dumps a full trace JSON to ``traces/`` for offline review.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from agents.aggregator import aggregate
from agents.devils_advocate import critique as devils_advocate
from agents.forecasters import forecast_both
from agents.planner import plan
from agents.synthesis import synthesize
from schemas import EventRequest, PredictionResponse, ResearchResult
from tools.base import import_tools, run_tool

log = logging.getLogger(__name__)

# Bring all tool modules into the registry exactly once
import_tools()

TRACE_DIR = Path(os.environ.get("TRACE_DIR", "traces"))


def _write_trace(market_ticker: str, trace: dict) -> Path | None:
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / f"{market_ticker}_{int(time.time())}.json"
        path.write_text(json.dumps(trace, indent=2, default=str))
        return path
    except Exception as exc:  # noqa: BLE001
        log.warning("trace dump failed: %s", exc)
        return None


async def _do_research(event: EventRequest, steps, per_tool_timeout_s: float = 90.0):
    coros = [
        run_tool(step.tool, event, step.args, timeout_s=per_tool_timeout_s)
        for step in steps
    ]
    results: list[ResearchResult] = await asyncio.gather(*coros, return_exceptions=False)
    for r in results:
        if r.error:
            log.warning("tool %s error: %s", r.tool, r.error)
        else:
            log.info("tool %s ok (%.1fs, %d items)", r.tool, r.elapsed_s, len(r.items))
    return results


async def run_pipeline(event: EventRequest) -> tuple[PredictionResponse, dict]:
    t0 = time.monotonic()
    trace: dict = {"event": event.model_dump(), "stages": {}}

    research_plan = await plan(event)
    trace["stages"]["plan"] = research_plan.model_dump()
    log.info(
        "plan: %s (%d steps)",
        research_plan.reasoning,
        len(research_plan.research_steps),
    )

    results = await _do_research(event, research_plan.research_steps)
    trace["stages"]["research"] = [r.model_dump() for r in results]

    brief = await synthesize(event, results)
    trace["stages"]["brief"] = brief
    log.info("brief: %d chars", len(brief))

    drafts = await forecast_both(event, brief)
    trace["stages"]["forecasts"] = [d.model_dump() for d in drafts]
    for d in drafts:
        log.info(
            "forecast %s: %s (conf=%s)",
            d.model,
            ", ".join(f"{p.market}={p.probability:.3f}" for p in d.probabilities),
            d.confidence,
        )

    critique_text = await devils_advocate(event, brief, drafts)
    trace["stages"]["critique"] = critique_text
    log.info("critique: %d chars", len(critique_text))

    final = await aggregate(event, brief, drafts, critique_text)
    trace["stages"]["final"] = final.model_dump()
    trace["total_s"] = time.monotonic() - t0

    log.info(
        "final %s: %s (total=%.2fs)",
        event.market_ticker,
        ", ".join(f"{p.market}={p.probability:.3f}" for p in final.probabilities),
        trace["total_s"],
    )

    path = _write_trace(event.market_ticker, trace)
    if path:
        log.info("trace written to %s", path)

    return final, trace
