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

GPT5 = "gpt-5"


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
) -> str:
    """Single-turn Claude completion. Returns concatenated text blocks."""
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if tools:
        kwargs["tools"] = tools
    response = await anthropic_client().messages.create(**kwargs)
    return "\n".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()


async def gpt_complete(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
) -> str:
    """Single-turn GPT completion. Returns None if OPENAI_API_KEY missing."""
    client = openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY not set")
    response = await client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()
