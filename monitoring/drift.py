"""
Drift monitoring — compares recent feature distributions against a reference dataset.

Uses Kolmogorov-Smirnov test and Population Stability Index (PSI) to detect
feature drift. Logs drift scores to MLflow. Can be run on a schedule or manually.

Replaces Evidently (incompatible with Python 3.14) with scipy-based detection.
"""

import sys
import os
import logging
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database.models import engine
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "drift-monitoring"

FEATURE_COLS = [
    "return_1", "return_5", "return_15",
    "vol_5", "vol_15",
    "volume_ratio",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_position",
]

# Thresholds
KS_WARN = 0.1    # yellow — moderate drift
KS_ALERT = 0.2   # red — significant drift
PSI_WARN = 0.1
PSI_ALERT = 0.2


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions."""
    # Create bins from reference
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Add small epsilon to avoid division by zero
    eps = 1e-6
    ref_pct = ref_counts / (ref_counts.sum() + eps) + eps
    cur_pct = cur_counts / (cur_counts.sum() + eps) + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def get_status(ks_stat: float, psi: float) -> str:
    """Return drift status: green, yellow, or red."""
    if ks_stat >= KS_ALERT or psi >= PSI_ALERT:
        return "red"
    if ks_stat >= KS_WARN or psi >= PSI_WARN:
        return "yellow"
    return "green"


def fetch_features(ticker: str, limit: int) -> pd.DataFrame:
    """Fetch feature rows for a ticker, ordered by time."""
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT " + ", ".join(FEATURE_COLS) + ", timestamp "
                "FROM features WHERE ticker = :ticker "
                "ORDER BY timestamp DESC LIMIT :limit"
            ),
            {"ticker": ticker, "limit": limit},
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=FEATURE_COLS + ["timestamp"])
    return df.sort_values("timestamp")


def run_drift_check(
    tickers: list[str] | None = None,
    reference_size: int = 500,
    current_size: int = 100,
) -> dict:
    """
    Compare the most recent `current_size` rows against the preceding
    `reference_size` rows for each ticker.

    Returns a dict with per-ticker, per-feature drift results.
    """
    if tickers is None:
        tickers = ["AAPL", "TSLA", "SPY"]

    all_results = {}

    for ticker in tickers:
        logger.info("Checking drift for %s ...", ticker)
        total_needed = reference_size + current_size
        df = fetch_features(ticker, total_needed)

        if len(df) < 50:
            logger.warning("%s: only %d rows, need at least 50 — skipping", ticker, len(df))
            continue

        # Split into reference (older) and current (newer)
        split_point = max(len(df) - current_size, 10)
        ref_df = df.iloc[:split_point]
        cur_df = df.iloc[split_point:]

        ticker_results = {}
        overall_status = "green"

        for col in FEATURE_COLS:
            ref_vals = ref_df[col].dropna().values
            cur_vals = cur_df[col].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
            psi = compute_psi(ref_vals, cur_vals)
            status = get_status(ks_stat, psi)

            if status == "red":
                overall_status = "red"
            elif status == "yellow" and overall_status == "green":
                overall_status = "yellow"

            ticker_results[col] = {
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_pval, 4),
                "psi": round(psi, 4),
                "status": status,
                "ref_mean": round(float(ref_vals.mean()), 4),
                "cur_mean": round(float(cur_vals.mean()), 4),
                "ref_std": round(float(ref_vals.std()), 4),
                "cur_std": round(float(cur_vals.std()), 4),
            }

        all_results[ticker] = {
            "features": ticker_results,
            "overall_status": overall_status,
            "reference_rows": len(ref_df),
            "current_rows": len(cur_df),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        drifted = [f for f, r in ticker_results.items() if r["status"] != "green"]
        logger.info(
            "%s: overall=%s | %d/%d features drifted",
            ticker, overall_status, len(drifted), len(ticker_results),
        )

    return all_results


def log_to_mlflow(results: dict):
    """Log drift results to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"drift-check-{datetime.now().strftime('%Y%m%d-%H%M')}"):
        for ticker, data in results.items():
            # Log overall status as a metric (green=0, yellow=1, red=2)
            status_map = {"green": 0, "yellow": 1, "red": 2}
            mlflow.log_metric(f"{ticker}_drift_status", status_map[data["overall_status"]])
            mlflow.log_metric(f"{ticker}_reference_rows", data["reference_rows"])
            mlflow.log_metric(f"{ticker}_current_rows", data["current_rows"])

            # Log per-feature KS stats
            for feature, fdata in data["features"].items():
                mlflow.log_metric(f"{ticker}_{feature}_ks", fdata["ks_statistic"])
                mlflow.log_metric(f"{ticker}_{feature}_psi", fdata["psi"])

        # Log full results as JSON artifact
        mlflow.log_text(json.dumps(results, indent=2), "drift_report.json")

    logger.info("Drift results logged to MLflow")


def main():
    """Run drift check and log results."""
    results = run_drift_check()
    if results:
        log_to_mlflow(results)
    else:
        logger.warning("No drift results to log.")


if __name__ == "__main__":
    main()
