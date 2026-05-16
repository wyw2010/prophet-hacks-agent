"""Sports odds via the-odds-api.com.

Free tier: 500 req/mo. Returns consensus implied probabilities from major books.
"""
from __future__ import annotations

import os

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register


ODDS_BASE = "https://api.the-odds-api.com/v4/sports"


SPORT_KEYS = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "soccer_epl": "soccer_epl",
}


@register(
    name="sports_odds",
    description=(
        "Live sportsbook odds, converted to implied probabilities. Use for any "
        "Sports event. The implied probability is the strongest available base rate."
    ),
    args_schema='{"sport": "nba", "team_query": "Cleveland", "regions": "us"}',
    requires_env=["ODDS_API_KEY"],
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    key = os.environ["ODDS_API_KEY"]
    sport = args.get("sport", "nba").lower()
    team_query = (args.get("team_query") or "").lower()
    regions = args.get("regions", "us")

    sport_key = SPORT_KEYS.get(sport, sport)

    async with httpx.AsyncClient(timeout=min(timeout_s, 20.0)) as client:
        try:
            r = await client.get(
                f"{ODDS_BASE}/{sport_key}/odds",
                params={
                    "apiKey": key,
                    "regions": regions,
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
            )
            r.raise_for_status()
            payload = r.json()
        except httpx.HTTPError as exc:
            return ResearchResult(tool="sports_odds", error=f"http error: {exc}")

    items: list[dict] = []
    for game in payload:
        home = (game.get("home_team") or "").lower()
        away = (game.get("away_team") or "").lower()
        if team_query and team_query not in home and team_query not in away:
            continue

        prices_by_team: dict[str, list[float]] = {}
        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    prices_by_team.setdefault(outcome["name"], []).append(
                        float(outcome["price"])
                    )

        implied = {}
        for team, decimals in prices_by_team.items():
            if not decimals:
                continue
            avg_decimal = sum(decimals) / len(decimals)
            implied[team] = round(1.0 / avg_decimal, 4)

        s = sum(implied.values())
        if s > 0:
            implied = {k: round(v / s, 4) for k, v in implied.items()}

        items.append(
            {
                "home": game.get("home_team"),
                "away": game.get("away_team"),
                "commence_time": game.get("commence_time"),
                "implied_probabilities": implied,
                "n_books": len(game.get("bookmakers", [])),
            }
        )

    summary = f"Matched {len(items)} games in {sport_key}; team_query={team_query!r}."
    return ResearchResult(tool="sports_odds", items=items, summary=summary)
