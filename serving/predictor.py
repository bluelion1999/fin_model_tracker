"""
Loads the production model from MLflow and scores feature vectors.
"""

import os
import logging

import mlflow
import numpy as np
import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database.models import engine

load_dotenv()

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTRY_NAME = "stock-direction-classifier"
PRODUCTION_ALIAS = "production"

FEATURE_COLS = [
    "return_1", "return_5", "return_15",
    "vol_5", "vol_15",
    "volume_ratio",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_position",
    "hour_of_day", "day_of_week",
]


class Predictor:
    def __init__(self):
        self.model = None
        self.model_version = None
        self.model_name = None
        self.load_model()

    def load_model(self):
        """Load the production model from MLflow registry."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTRY_NAME}@{PRODUCTION_ALIAS}"
        logger.info("Loading model from %s", model_uri)

        self.model = mlflow.pyfunc.load_model(model_uri)
        # Extract version info
        client = mlflow.tracking.MlflowClient()
        version_info = client.get_model_version_by_alias(REGISTRY_NAME, PRODUCTION_ALIAS)
        self.model_version = version_info.version
        self.model_name = REGISTRY_NAME
        logger.info("Loaded model %s version %s", self.model_name, self.model_version)

    def get_latest_features(self, ticker: str) -> dict | None:
        """Fetch the most recent feature row for a ticker from the DB."""
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT timestamp, " + ", ".join(FEATURE_COLS) +
                    " FROM features WHERE ticker = :ticker "
                    "ORDER BY timestamp DESC LIMIT 1"
                ),
                {"ticker": ticker},
            )
            row = result.fetchone()

        if row is None:
            return None

        data = {"timestamp": row[0]}
        for i, col in enumerate(FEATURE_COLS):
            data[col] = row[i + 1]
        return data

    def predict(self, ticker: str) -> dict | None:
        """Get prediction for a ticker using latest features."""
        features = self.get_latest_features(ticker)
        if features is None:
            return None

        feature_values = np.array([[features[col] for col in FEATURE_COLS]])
        df_input = pd.DataFrame(feature_values, columns=FEATURE_COLS)

        # Get prediction and probability
        prediction = int(self.model.predict(df_input)[0])

        # Extract probability from the underlying model
        probability = None
        impl = self.model._model_impl
        raw_model = getattr(impl, "xgb_model", None) or getattr(impl, "sklearn_model", None)
        if raw_model is not None and hasattr(raw_model, "predict_proba"):
            proba = raw_model.predict_proba(df_input)
            probability = float(proba[0][1])

        return {
            "ticker": ticker,
            "timestamp": features["timestamp"].isoformat(),
            "predicted_label": prediction,
            "probability": probability,
            "direction": "UP" if prediction == 1 else "DOWN",
            "model_version": self.model_version,
        }

    def model_info(self) -> dict:
        """Return metadata about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "production_alias": PRODUCTION_ALIAS,
            "features": FEATURE_COLS,
        }
