from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
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


def main() -> None:
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)


if __name__ == "__main__":
    main()
