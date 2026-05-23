from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from pipeline import run_pipeline
from schemas import EventRequest, PredictionResponse

load_dotenv(override=True)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("agent")

app = FastAPI(title="Prophet Hacks Forecasting Agent")

TRACE_DIR = Path(os.environ.get("TRACE_DIR", "traces"))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(event: EventRequest, debug: bool = False):
    """Forecast probabilities for an event.

    Query param ``?debug=1`` returns the full pipeline trace alongside the
    prediction. Without it, returns the standard PredictionResponse so the
    harness contract is unchanged.
    """
    log.info("predict event=%s title=%r debug=%s", event.market_ticker, event.title, debug)
    response, trace = await run_pipeline(event)
    if debug:
        return JSONResponse({"prediction": response.model_dump(), "trace": trace})
    return response


# ---------------------------------------------------------------------------
# Trace inspection endpoints
#
# Required env var: ``TRACE_ACCESS_TOKEN``. Both endpoints 503 if it's not
# set (so traces are NEVER public by default), and 403 on a bad token.
# Call with ``?token=...`` query param.
# ---------------------------------------------------------------------------

def _check_token(token: str | None) -> None:
    expected = os.environ.get("TRACE_ACCESS_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="TRACE_ACCESS_TOKEN not configured; trace endpoints disabled",
        )
    if token != expected:
        raise HTTPException(status_code=403, detail="invalid token")


def _safe_trace_path(name: str) -> Path:
    """Resolve a trace filename to a real path under TRACE_DIR. Rejects
    anything that tries to escape via .. or absolute paths."""
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid trace name")
    target = (TRACE_DIR / name).resolve()
    base = TRACE_DIR.resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="path escapes trace dir")
    return target


@app.get("/traces")
async def list_traces(token: str | None = None, limit: int = 200):
    """List trace files (most recent first)."""
    _check_token(token)
    if not TRACE_DIR.exists():
        return {"count": 0, "traces": []}

    entries = []
    for f in TRACE_DIR.glob("*.json"):
        stat = f.stat()
        entries.append(
            {
                "name": f.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    entries.sort(key=lambda e: e["modified"], reverse=True)
    return {"count": len(entries), "traces": entries[:limit]}


@app.get("/traces/{name}")
async def get_trace(name: str, token: str | None = None):
    """Return a single trace JSON by filename."""
    _check_token(token)
    target = _safe_trace_path(name)
    if not target.exists():
        raise HTTPException(status_code=404, detail="trace not found")
    try:
        return JSONResponse(json.loads(target.read_text()))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"trace not valid JSON: {exc}")


def main() -> None:
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)


if __name__ == "__main__":
    main()
