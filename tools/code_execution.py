"""Anthropic native Python code execution via the ``code_execution_20250522`` tool.

The model writes and runs Python in Anthropic's sandbox — useful for arithmetic
on the brief (de-vigging implied probabilities, base-rate calculations, monte
carlo, normalizing distributions). Returns the final text plus any code that
was executed for trace visibility.

Bundled with the same Anthropic API key as the rest of the pipeline; no
separate auth.
"""
from __future__ import annotations

import anthropic

from llm_clients import SONNET, anthropic_client
from schemas import EventRequest, ResearchResult
from tools.base import register


SYSTEM = """You are a quantitative assistant for a forecasting agent. Use the \
code execution tool to perform precise calculations relevant to the forecasting \
question — de-vig odds, compute base rates, normalize distributions, simulate \
scenarios, parse messy numbers. Return key findings as bullets with numeric \
results. Show the numbers, not the reasoning."""


@register(
    name="code_execution",
    description=(
        "Anthropic-hosted Python code execution. Use for precise math: "
        "de-vigging sportsbook implied probabilities, base-rate calculations, "
        "monte carlo simulation, normalizing distributions, parsing "
        "numeric data in the brief. Pass a `task` describing what to compute."
    ),
    args_schema=(
        '{"task": "De-vig the NBA championship odds: OKC -180, SA +350, '
        'NYK +600, DET +1800, CLE +4500. Return implied probabilities '
        'normalized to sum to 1.", "max_uses": 3}'
    ),
    requires_env=["ANTHROPIC_API_KEY"],
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    task = args.get("task") or f"Quantitative analysis for: {event.title}"
    max_uses = int(args.get("max_uses", 3))

    user = (
        f"Forecasting question: {event.title}\n"
        f"Category: {event.category}\n\n"
        f"Task: {task}\n\n"
        f"Use the code execution tool up to {max_uses} times. Return 3-8 "
        "bulleted findings with concrete numeric results."
    )

    try:
        response = await anthropic_client().messages.create(
            model=SONNET,
            max_tokens=2048,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "type": "code_execution_20250522",
                    "name": "code_execution",
                }
            ],
            extra_headers={"anthropic-beta": "code-execution-2025-05-22"},
        )
    except anthropic.APIError as exc:
        return ResearchResult(tool="code_execution", error=f"anthropic api: {exc}")

    text_blocks = [
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ]
    summary = "\n".join(text_blocks).strip()

    executed_code: list[dict] = []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype in ("server_tool_use", "tool_use"):
            block_input = getattr(block, "input", {}) or {}
            if isinstance(block_input, dict) and block_input.get("code"):
                executed_code.append({"code": block_input["code"]})

    return ResearchResult(
        tool="code_execution",
        items=executed_code,
        summary=summary,
    )
