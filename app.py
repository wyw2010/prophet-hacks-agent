from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI

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


@app.post("/predict", response_model=PredictionResponse)
async def predict(event: EventRequest) -> PredictionResponse:
    log.info("predict event=%s title=%r", event.market_ticker, event.title)
    return await run_pipeline(event)


def main() -> None:
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)


if __name__ == "__main__":
    main()
