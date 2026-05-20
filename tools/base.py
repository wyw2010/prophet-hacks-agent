"""Uniform interface for all research tools.

Each tool is an async callable with signature:
    async def run(event, args, timeout_s) -> ResearchResult

Tools self-register via the `@register` decorator. The planner sees only tools
whose `available()` predicate returns True (i.e., the API key is set), so missing
keys silently disable that tool rather than crashing the pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from schemas import EventRequest, ResearchResult

log = logging.getLogger(__name__)

ToolFn = Callable[[EventRequest, dict, float], Awaitable[ResearchResult]]


@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: str
    available: Callable[[], bool]
    run: ToolFn


TOOL_REGISTRY: dict[str, ToolSpec] = {}


def register(
    *,
    name: str,
    description: str,
    args_schema: str,
    requires_env: list[str] | None = None,
):
    """Decorator to register a tool. `requires_env` is a list of env-var names;
    if any are missing the tool is hidden from the planner."""

    def _decorator(fn: ToolFn) -> ToolFn:
        env_keys = requires_env or []

        def _available() -> bool:
            return all(os.environ.get(k) for k in env_keys)

        TOOL_REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            args_schema=args_schema,
            available=_available,
            run=fn,
        )
        return fn

    return _decorator


def available_tools() -> list[ToolSpec]:
    return [t for t in TOOL_REGISTRY.values() if t.available()]


def TOOL_CATALOG() -> str:
    """Render available tools as a catalog string for the planner prompt."""
    lines = []
    for t in available_tools():
        lines.append(f"- {t.name}: {t.description}")
        lines.append(f"  args: {t.args_schema}")
    return "\n".join(lines) if lines else "(no tools available)"


async def run_tool(
    name: str,
    event: EventRequest,
    args: dict,
    timeout_s: float = 90.0,
) -> ResearchResult:
    """Invoke a tool by name with a hard timeout and exception isolation."""
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return ResearchResult(tool=name, error=f"unknown tool: {name}")
    if not spec.available():
        return ResearchResult(tool=name, error="tool unavailable (missing env var)")

    start = time.monotonic()
    try:
        result = await asyncio.wait_for(spec.run(event, args, timeout_s), timeout=timeout_s)
        result.elapsed_s = time.monotonic() - start
        return result
    except asyncio.TimeoutError:
        return ResearchResult(
            tool=name, error=f"timeout after {timeout_s}s", elapsed_s=time.monotonic() - start
        )
    except Exception as exc:  # noqa: BLE001 — tool errors must not crash pipeline
        log.exception("tool %s raised", name)
        return ResearchResult(
            tool=name, error=f"{type(exc).__name__}: {exc}", elapsed_s=time.monotonic() - start
        )


def import_tools() -> None:
    """Import every tool module so its @register decorator runs.

    Called once at startup; modules can also be imported individually for tests.

    NOTE: ``polymarket_related`` and ``wikipedia`` are intentionally NOT imported
    here. Empirically they returned 0 items on 84% / 57% of the eval batch
    respectively, so we hide them from the planner to reduce noise and pipeline
    latency. The modules still exist on disk and can be re-imported individually
    if needed.
    """
    from tools import (  # noqa: F401
        claude_news,
        code_execution,
        congress,
        court_docket,
        crypto,
        earnings_data,
        fred,
        kalshi_related,
        mengye_search,
        sports,
    )
