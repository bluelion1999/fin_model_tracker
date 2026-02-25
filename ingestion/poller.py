"""
Fetches 1-minute OHLCV data for AAPL, TSLA, SPY every 60 seconds
and stores it in the raw_prices table.
"""

import sys
import os
import logging
from datetime import datetime, timezone

import yfinance as yf
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database.models import engine, SessionLocal, RawPrice
from ingestion.features import compute_and_store_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TICKERS = ["AAPL", "TSLA", "SPY"]


def fetch_and_store():
    """Fetch the latest 1-min bars and upsert into raw_prices."""
    logger.info("Polling %s ...", TICKERS)

    for ticker in TICKERS:
        try:
            # Fetch last 1 day of 1-min data (yfinance minimum window for 1m interval)
            df = yf.download(ticker, period="1d", interval="1m", progress=False)

            if df.empty:
                logger.warning("No data returned for %s", ticker)
                continue

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            ts_col = "Datetime" if "Datetime" in df.columns else "index"
            df = df.rename(columns={ts_col: "timestamp"})

            rows = []
            for _, row in df.iterrows():
                rows.append(
                    {
                        "ticker": ticker,
                        "timestamp": row["timestamp"],
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]),
                    }
                )

            if not rows:
                continue

            # Upsert: insert new rows, skip duplicates
            with engine.begin() as conn:
                stmt = insert(RawPrice.__table__).values(rows)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["ticker", "timestamp"]
                )
                result = conn.execute(stmt)
                logger.info(
                    "%s: fetched %d bars, inserted %d new",
                    ticker,
                    len(rows),
                    result.rowcount,
                )

            # Compute features for this ticker after ingesting new data
            compute_and_store_features(ticker)

        except Exception:
            logger.exception("Error fetching %s", ticker)


def main():
    logger.info("Starting ingestion poller for %s", TICKERS)

    # Run once immediately
    fetch_and_store()

    # Then schedule every 60 seconds
    scheduler = BlockingScheduler()
    scheduler.add_job(fetch_and_store, "interval", seconds=60)
    logger.info("Scheduler started — polling every 60s. Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Poller stopped.")


if __name__ == "__main__":
    main()
