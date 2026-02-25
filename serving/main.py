"""
FastAPI inference server.

Endpoints:
  GET  /health      — Liveness check
  GET  /model-info  — Model metadata
  GET  /predict/{ticker} — Prediction for a ticker using latest features
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from serving.predictor import Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VALID_TICKERS = {"AAPL", "TSLA", "SPY"}

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading production model...")
    predictor = Predictor()
    logger.info("Model loaded — server ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Finance ML Prediction API",
    description="Predicts stock direction (up/down in 15 min) for AAPL, TSLA, SPY",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.get("/model-info")
def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor.model_info()


@app.get("/predict/{ticker}")
def predict(ticker: str):
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker '{ticker}'. Must be one of {sorted(VALID_TICKERS)}",
        )

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = predictor.predict(ticker)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No features available for {ticker}. Run the ingestion poller first.",
        )

    return result
