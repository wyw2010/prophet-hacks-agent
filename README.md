# Prophet Hacks 2026 — Forecasting Agent

A calibrated forecasting agent for the [Prophet Arena](https://prophetarena.co) hackathon (Forecasting Track). Exposes a `POST /predict` endpoint that the evaluation harness queries with binary-outcome events; returns calibrated probabilities.

## Architecture

```
event JSON
   ↓
[ planner (Opus 4.7) ]            chooses which research tools to invoke
   ↓
[ parallel research ]             news, related markets, FRED, sports odds, ...
   ↓
[ synthesis (Sonnet) ]            structured evidence brief
   ↓
[ parallel forecasts ]            Opus 4.7 + GPT-5, same prompt
   ↓
[ devil's advocate (Opus) ]       red-team critique
   ↓
[ aggregator (Opus) ]             final calibrated distribution
   ↓
PredictionResponse
```

## Local quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add at minimum ANTHROPIC_API_KEY
uvicorn app:app --reload --port 8000
```

Test against the official harness:

```bash
# from the ai-prophet repo:
prophet forecast retrieve -o events.json
prophet forecast predict --events events.json --agent-url http://localhost:8000/predict
```

## Research tools

Each tool self-registers and only appears to the planner when its env vars are present:

| Tool | Env var | Notes |
|---|---|---|
| `mengye_search` | `MENGYE_SEARCH_URL` | Custom semantic news search |
| `claude_news` | `ANTHROPIC_API_KEY` | Anthropic native web search |
| `kalshi_related` | — | Related Kalshi markets (excludes self) |
| `polymarket_related` | — | Polymarket Gamma API |
| `wikipedia` | — | Wikipedia REST summaries |
| `fred` | `FRED_API_KEY` | US economic data |
| `congress_bills` | `CONGRESS_API_KEY` | Federal legislation |
| `sports_odds` | `ODDS_API_KEY` | Vegas implied probabilities |
| `crypto` | — | CoinGecko spot + history |

## Endpoint contract

`POST /predict` accepts an [EventRequest](schemas.py) and returns:

```json
{"probabilities": [{"market": "<outcome>", "probability": 0.68}, ...]}
```

Probabilities sum to 1.0; outcome labels must match `event.outcomes` exactly.

## Deployment

`render.yaml` is configured for Render. Set the env vars in the dashboard (they're listed in `.env.example`), connect this GitHub repo, deploy. The `/health` endpoint is the platform health check.
