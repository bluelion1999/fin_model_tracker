"""
Compute technical indicator features from raw_prices and store in the features table.

Features computed per ticker:
  - Log returns: 1-bar, 5-bar, 15-bar
  - Volatility: Rolling std of 1-bar returns (5, 15 windows)
  - Volume ratio: Current volume / 20-bar rolling average volume
  - RSI: 14-period relative strength index
  - MACD: 12/26 EMA diff, signal line (9-period EMA of MACD), histogram
  - Bollinger Band position: (close - lower) / (upper - lower)
  - Time features: hour of day, day of week
  - Label: 1 if close[t+15] > close[t], else 0 (NULL for recent rows)
"""

import sys
import os
import logging

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database.models import engine, Feature

logger = logging.getLogger(__name__)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given a DataFrame with OHLCV columns, compute all features.

    Expects columns: timestamp, open, high, low, close, volume
    Returns DataFrame with feature columns added.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    close = df["close"]
    volume = df["volume"].astype(float)

    # --- Log returns ---
    df["return_1"] = np.log(close / close.shift(1))
    df["return_5"] = np.log(close / close.shift(5))
    df["return_15"] = np.log(close / close.shift(15))

    # --- Volatility (rolling std of 1-bar returns) ---
    df["vol_5"] = df["return_1"].rolling(5).std()
    df["vol_15"] = df["return_1"].rolling(15).std()

    # --- Volume ratio ---
    vol_ma_20 = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_ma_20

    # --- RSI (14-period) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- Bollinger Band position ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = bb_upper - bb_lower
    df["bb_position"] = np.where(bb_range > 0, (close - bb_lower) / bb_range, 0.5)

    # --- Time features ---
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # --- Label: 1 if close[t+15] > close[t], else 0 ---
    future_close = close.shift(-15)
    df["label"] = np.where(future_close > close, 1, 0)
    # Set label to NaN for the last 15 rows (future unknown)
    df.loc[df.index[-15:], "label"] = None

    return df


def compute_and_store_features(ticker: str):
    """Pull raw prices for a ticker, compute features, upsert into features table."""
    logger.info("Computing features for %s", ticker)

    with engine.begin() as conn:
        result = conn.execute(
            text(
                "SELECT timestamp, open, high, low, close, volume "
                "FROM raw_prices WHERE ticker = :ticker "
                "ORDER BY timestamp"
            ),
            {"ticker": ticker},
        )
        rows = result.fetchall()

    if len(rows) < 30:
        logger.warning("%s: only %d bars, need at least 30 for features — skipping", ticker, len(rows))
        return

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = compute_features(df)

    # Drop rows where core features are NaN (warmup period)
    feature_cols = [
        "return_1", "return_5", "return_15", "vol_5", "vol_15",
        "volume_ratio", "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_position",
    ]
    df_valid = df.dropna(subset=feature_cols).copy()

    if df_valid.empty:
        logger.warning("%s: no valid feature rows after dropping NaN", ticker)
        return

    records = []
    for _, row in df_valid.iterrows():
        record = {"ticker": ticker, "timestamp": row["timestamp"]}
        for col in feature_cols + ["hour_of_day", "day_of_week"]:
            record[col] = float(row[col]) if pd.notna(row[col]) else None
        record["label"] = int(row["label"]) if pd.notna(row["label"]) else None
        records.append(record)

    with engine.begin() as conn:
        stmt = insert(Feature.__table__).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "timestamp"],
            set_={col: stmt.excluded[col] for col in feature_cols + ["hour_of_day", "day_of_week", "label"]},
        )
        conn.execute(stmt)

    logger.info("%s: upserted %d feature rows", ticker, len(records))
