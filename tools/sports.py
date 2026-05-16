"""Sports odds via the-odds-api.com.

Free tier: 500 req/mo. Returns implied probabilities from major books.

The Odds API supports two relevant market types:
  - ``h2h`` (head-to-head): a single scheduled game between two teams.
  - ``outrights``: tournament/championship futures (one event, N outcomes).

Sport keys with suffixes like ``_championship_winner`` are futures and require
``markets=outrights``. Regular sport keys (``basketball_nba``, ``soccer_epl``)
default to ``h2h``. Planner can override with an explicit ``markets`` arg.
"""
from __future__ import annotations

import os

import httpx

from schemas import EventRequest, ResearchResult
from tools.base import register


ODDS_BASE = "https://api.the-odds-api.com/v4/sports"

# Convenience aliases — planner can pass short keys.
SPORT_KEYS = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "soccer_epl": "soccer_epl",
}


FUTURES_SUFFIXES = (
    "_championship_winner",
    "_winner",
    "_outright",
    "_outrights",
)


def _is_futures(sport_key: str) -> bool:
    return any(sport_key.endswith(s) for s in FUTURES_SUFFIXES)


@register(
    name="sports_odds",
    description=(
        "Live sportsbook odds, converted to implied probabilities. Use for any "
        "Sports event. Pass a sport key like 'basketball_nba' for individual "
        "games OR 'basketball_nba_championship_winner' for tournament futures. "
        "Tournament/championship sport keys auto-select markets=outrights."
    ),
    args_schema=(
        '{"sport": "basketball_nba_championship_winner", '
        '"team_query": "", "markets": "outrights", "regions": "us"}'
    ),
    requires_env=["ODDS_API_KEY"],
)
async def run(event: EventRequest, args: dict, timeout_s: float) -> ResearchResult:
    key = os.environ["ODDS_API_KEY"]
    sport = args.get("sport", "nba").lower()
    team_query = (args.get("team_query") or "").lower()
    regions = args.get("regions", "us")

    sport_key = SPORT_KEYS.get(sport, sport)
    markets = args.get("markets") or ("outrights" if _is_futures(sport_key) else "h2h")

    async with httpx.AsyncClient(timeout=min(timeout_s, 20.0)) as client:
        try:
            r = await client.get(
                f"{ODDS_BASE}/{sport_key}/odds",
                params={
                    "apiKey": key,
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": "decimal",
                },
            )
            r.raise_for_status()
            payload = r.json()
        except httpx.HTTPError as exc:
            return ResearchResult(
                tool="sports_odds",
                error=f"http error ({sport_key}, markets={markets}): {exc}",
            )

    if markets == "outrights":
        items = _parse_outrights(payload, team_query)
    else:
        items = _parse_h2h(payload, team_query)

    summary = f"{sport_key} markets={markets}: {len(items)} entries; team_query={team_query!r}."
    return ResearchResult(tool="sports_odds", items=items, summary=summary)


def _parse_h2h(payload: list[dict], team_query: str) -> list[dict]:
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

        implied: dict[str, float] = {}
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
    return items


def _parse_outrights(payload: list[dict], team_query: str) -> list[dict]:
    """Outrights: one tournament event, N outcomes (teams)."""
    implied_by_team: dict[str, list[float]] = {}
    n_books = 0
    sport_title = ""

    for event in payload:
        sport_title = event.get("sport_title") or sport_title
        for book in event.get("bookmakers", []):
            n_books += 1
            for market in book.get("markets", []):
                if market.get("key") not in ("outrights", "outrights_lay"):
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name") or ""
                    try:
                        price = float(outcome.get("price"))
                    except (TypeError, ValueError):
                        continue
                    if price <= 0:
                        continue
                    implied_by_team.setdefault(name, []).append(1.0 / price)

    # Average across books, then normalize (removes vig).
    averaged: dict[str, float] = {}
    for team, ps in implied_by_team.items():
        if ps:
            averaged[team] = sum(ps) / len(ps)

    total = sum(averaged.values()) or 1.0
    items: list[dict] = []
    for team, p in sorted(averaged.items(), key=lambda kv: -kv[1]):
        if team_query and team_query not in team.lower():
            continue
        items.append(
            {
                "team": team,
                "implied_probability": round(p / total, 4),
                "raw_avg_implied": round(p, 4),
            }
        )
    if not items:
        return []
    return [
        {
            "sport_title": sport_title,
            "n_books": n_books,
            "outright_probabilities": items,
        }
    ]
