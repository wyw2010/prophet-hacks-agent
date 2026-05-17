"""Unit-test each research tool locally without going through the LLM pipeline.

Hits the actual external APIs (Odds API, Kalshi, Polymarket, Wikipedia, FRED,
CoinGecko, Mengye, Anthropic web search) with realistic args. Useful for fast
verification after tool-level changes — no Anthropic/OpenAI cost.

Usage (from prophet-hacks-agent/):
    python scripts/verify_tools.py [tool_name]
    # tool_name optional; runs all tools if omitted
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Repo root on sys.path so tool modules can import schemas/tools.base
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

from schemas import EventRequest
from tools.base import TOOL_REGISTRY, import_tools, run_tool

import_tools()


FAKE_NBA = EventRequest(
    event_ticker="KXNBA-26",
    market_ticker="KXNBA-26",
    title="Who will win the 2026 NBA Finals?",
    category="Sports",
    close_time="2026-06-30T14:00:00Z",
    outcomes=["Oklahoma City", "San Antonio", "New York", "Cleveland"],
)

FAKE_FED = EventRequest(
    event_ticker="KXFED-26SEP",
    market_ticker="KXFED-26SEP",
    title="Where will the upper bound of the US federal funds target rate sit after the September 2026 FOMC meeting?",
    category="Economics",
    close_time="2026-09-16T17:55:00Z",
    outcomes=["Above 3.50%", "Above 3.75%", "Above 4.00%"],
)

FAKE_EMMY = EventRequest(
    event_ticker="KXEMMYLSERIES-26SEP14",
    market_ticker="KXEMMYLSERIES-26SEP14",
    title="Which show will win Outstanding Limited or Anthology Series at the 78th Emmy Awards?",
    category="Entertainment",
    close_time="2026-09-14T22:00:00Z",
    outcomes=["Beef", "Love Story", "Half Man"],
)


CASES: list[tuple[str, EventRequest, dict]] = [
    # The critical fix: futures sport key + outrights market
    (
        "sports_odds",
        FAKE_NBA,
        {"sport": "basketball_nba_championship_winner", "regions": "us"},
    ),
    # Wikipedia action API (was 403'ing on REST)
    (
        "wikipedia",
        FAKE_EMMY,
        {"titles": ["Beef (TV series)", "Bank of Japan", "Oklahoma City Thunder"]},
    ),
    # Polymarket: sorted by volume, broader matching
    (
        "polymarket_related",
        FAKE_FED,
        {"keywords": ["fed", "rate", "fomc", "interest rate"], "limit_per_keyword": 5},
    ),
    # Kalshi: volume filter should skip the zero-volume noise
    (
        "kalshi_related",
        FAKE_FED,
        {"series_tickers": ["KXFED"], "keywords": ["fed rate"], "limit_per_source": 10},
    ),
    # FRED — known working, included for completeness
    (
        "fred",
        FAKE_FED,
        {"series_ids": ["DFEDTARU", "CPIAUCSL", "UNRATE"], "lookback_days": 180},
    ),
    # CoinGecko (no key required)
    (
        "crypto",
        FAKE_NBA,
        {"asset_ids": ["bitcoin", "ethereum"], "history_days": 30},
    ),
    # Mengye search server
    (
        "mengye_search",
        FAKE_NBA,
        {"queries": ["NBA Finals 2026 odds"], "lookback_days": 30, "top_k": 5},
    ),
    # NEW: Anthropic code execution
    (
        "code_execution",
        FAKE_NBA,
        {
            "task": (
                "De-vig the NBA championship odds: OKC -180, SA +350, NYK +600, "
                "DET +1800, CLE +4500. Return implied probabilities normalized to "
                "sum to 1."
            ),
            "max_uses": 2,
        },
    ),
    # NEW: CourtListener search
    (
        "court_docket",
        FAKE_EMMY,
        {"query": "Trump immunity", "type": "o", "limit": 5},
    ),
    # NEW: Yahoo Finance
    (
        "earnings_data",
        FAKE_NBA,
        {"tickers": ["AAPL", "NVDA"], "history_days": 30},
    ),
]


def short(s: str, n: int = 200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "..."


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", nargs="?", help="Only run this tool (optional)")
    args = parser.parse_args()

    print(f"Registered tools ({len(TOOL_REGISTRY)}): {sorted(TOOL_REGISTRY)}\n")

    for tool_name, event, tool_args in CASES:
        if args.tool and tool_name != args.tool:
            continue

        spec = TOOL_REGISTRY.get(tool_name)
        if spec is None:
            print(f"  {tool_name}: NOT REGISTERED — skipping")
            continue
        if not spec.available():
            print(f"  {tool_name}: not available (missing env var) — skipping\n")
            continue

        print(f"### {tool_name}")
        print(f"  event: {event.title[:60]}")
        print(f"  args:  {tool_args}")
        result = await run_tool(tool_name, event, tool_args, timeout_s=60.0)
        if result.error:
            print(f"  ❌ ERROR ({result.elapsed_s:.1f}s): {result.error}")
        else:
            print(f"  ✅ OK ({result.elapsed_s:.1f}s)  items={len(result.items)}")
            print(f"  summary: {short(result.summary, 250)}")
            if result.items:
                print(f"  sample item: {short(json.dumps(result.items[0], default=str), 400)}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
