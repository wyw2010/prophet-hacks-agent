"""Thin async wrappers over Anthropic and OpenAI SDKs.

Each helper returns the assistant text directly. JSON parsing is left to callers
so they can keep model-specific quirks isolated.
"""
from __future__ import annotations

import os
from functools import lru_cache

import anthropic
from openai import AsyncOpenAI


OPUS = "claude-opus-4-7"
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"

GPT = "gpt-5.5"

# Reasoning effort for GPT-5.5 calls. Valid: "none" | "low" | "medium" | "high" | "xhigh".
# Default lowered from "high" to "low" for cost control — see also CLAUDE_EFFORT.
GPT_DEFAULT_REASONING = os.environ.get("GPT_REASONING", "low")

# Extended-thinking effort for Claude calls (Claude 4.x adaptive-thinking API).
# Valid: "none" | "minimal" | "low" | "medium" | "high" (passed as output_config.effort).
# Default lowered from "high" to "low" — high mode burned ~$1/call in thinking tokens
# across 3 Opus stages. "low" keeps reasoning quality while cutting cost ~75%.
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "low")


@lru_cache(maxsize=1)
def anthropic_client() -> anthropic.AsyncAnthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.AsyncAnthropic(api_key=key)


@lru_cache(maxsize=1)
def openai_client() -> AsyncOpenAI | None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    return AsyncOpenAI(api_key=key)


async def claude_complete(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    tools: list[dict] | None = None,
    effort: str | None = None,
) -> str:
    """Single-turn Claude completion. Returns concatenated final-text blocks.

    If ``effort`` is anything other than ``None``/``"none"``, extended thinking
    is enabled via ``thinking.type=adaptive`` and ``output_config.effort`` is
    set. Streaming is used because high-effort requests can exceed the SDK's
    non-streaming time cap. The model's internal reasoning blocks are
    discarded; only the final assistant text is returned.
    """
    effort_norm = (effort or "none").lower()
    thinking_on = effort_norm not in ("none", "")

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if tools:
        kwargs["tools"] = tools
    if thinking_on:
        kwargs["thinking"] = {"type": "adaptive"}
        # output_config is newer than the installed SDK's typed signature;
        # extra_body passes it through to the raw request body.
        kwargs["extra_body"] = {"output_config": {"effort": effort_norm}}

    client = anthropic_client()
    if thinking_on:
        async with client.messages.stream(**kwargs) as stream:
            final = await stream.get_final_message()
        content = final.content
    else:
        response = await client.messages.create(**kwargs)
        content = response.content

    return "\n".join(
        block.text for block in content if getattr(block, "type", None) == "text"
    ).strip()


async def gpt_complete(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 8000,
    reasoning_effort: str | None = None,
) -> str:
    """Single-turn GPT completion.

    For gpt-5.5 with reasoning enabled, the response includes hidden reasoning
    tokens; bump ``max_tokens`` accordingly (default 8000 leaves room).
    """
    client = openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY not set")

    kwargs: dict = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    effort = reasoning_effort or GPT_DEFAULT_REASONING
    if effort and effort != "none":
        kwargs["reasoning_effort"] = effort

    response = await client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()
