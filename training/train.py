"""
Offline training pipeline:
  1. Pull 2 years of historical 1-hour data via yfinance
  2. Compute features (same as ingestion/features.py)
  3. Train RandomForestClassifier and XGBClassifier
  4. Log runs to MLflow, register the better model as Production
"""

import sys
import os
import logging

import numpy as np
import pandas as pd
import yfinance as yf
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ingestion.features import compute_features
from training.evaluate import compute_metrics, print_evaluation
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TICKERS = ["AAPL", "TSLA", "SPY"]
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "stock-direction-15min"

FEATURE_COLS = [
    "return_1", "return_5", "return_15",
    "vol_5", "vol_15",
    "volume_ratio",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_position",
    "hour_of_day", "day_of_week",
]


def fetch_historical_data() -> pd.DataFrame:
    """Download 2 years of hourly data for all tickers and compute features."""
    all_frames = []

    for ticker in TICKERS:
        logger.info("Downloading historical data for %s ...", ticker)
        df = yf.download(ticker, period="2y", interval="1h", progress=False)

        if df.empty:
            logger.warning("No historical data for %s", ticker)
            continue

        # Flatten multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        ts_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(columns={
            ts_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        df["ticker"] = ticker
        df = compute_features(df)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info("Total rows before cleanup: %d", len(combined))

    # Drop rows missing features or labels
    combined = combined.dropna(subset=FEATURE_COLS + ["label"])
    combined["label"] = combined["label"].astype(int)
    logger.info("Total rows after cleanup: %d", len(combined))

    return combined


def time_based_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """Split data by time — last test_ratio of each ticker goes to test."""
    train_frames, test_frames = [], []

    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].sort_values("timestamp")
        split_idx = int(len(ticker_df) * (1 - test_ratio))
        train_frames.append(ticker_df.iloc[:split_idx])
        test_frames.append(ticker_df.iloc[split_idx:])

    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)

    return train, test


def train_model(model, model_name: str, X_train, y_train, X_test, y_test, params: dict):
    """Train a model, evaluate, log to MLflow, return metrics."""
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("features", FEATURE_COLS)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        logger.info("Training %s ...", model_name)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = print_evaluation(y_test, y_pred, y_prob, label=model_name)
        mlflow.log_metrics(metrics)

        # Log label distribution
        train_pos_rate = y_train.mean()
        test_pos_rate = y_test.mean()
        mlflow.log_metric("train_positive_rate", train_pos_rate)
        mlflow.log_metric("test_positive_rate", test_pos_rate)
        logger.info("Label dist — train: %.2f%% positive, test: %.2f%% positive",
                     train_pos_rate * 100, test_pos_rate * 100)

        # Log model
        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        logger.info("%s run_id: %s  |  AUC: %.4f  |  F1: %.4f",
                     model_name, run_id, metrics["auc"], metrics["f1"])

    return metrics, run_id


def register_best_model(best_name: str, best_run_id: str):
    """Register the best model in MLflow Model Registry."""
    model_uri = f"runs:/{best_run_id}/model"
    registry_name = "stock-direction-classifier"

    result = mlflow.register_model(model_uri, registry_name)
    logger.info("Registered model '%s' version %s from %s",
                registry_name, result.version, best_name)

    # Set alias to "production"
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(registry_name, "production", result.version)
    logger.info("Model version %s aliased as 'production'", result.version)


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Fetch and prepare data
    df = fetch_historical_data()
    train_df, test_df = time_based_split(df)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label"].values

    logger.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))

    # 2. Train RandomForest
    rf_params = {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 20, "random_state": 42}
    rf = RandomForestClassifier(**rf_params, n_jobs=-1)
    rf_metrics, rf_run_id = train_model(rf, "RandomForest", X_train, y_train, X_test, y_test, rf_params)

    # 3. Train XGBoost
    xgb_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
    }
    xgb = XGBClassifier(**xgb_params, n_jobs=-1)
    xgb_metrics, xgb_run_id = train_model(xgb, "XGBoost", X_train, y_train, X_test, y_test, xgb_params)

    # 4. Register the better model
    if rf_metrics["auc"] >= xgb_metrics["auc"]:
        logger.info("RandomForest wins (AUC %.4f vs %.4f)", rf_metrics["auc"], xgb_metrics["auc"])
        register_best_model("RandomForest", rf_run_id)
    else:
        logger.info("XGBoost wins (AUC %.4f vs %.4f)", xgb_metrics["auc"], rf_metrics["auc"])
        register_best_model("XGBoost", xgb_run_id)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
