"""
Shared database queries for the Streamlit dashboard.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'finance')}"
    f":{os.getenv('POSTGRES_PASSWORD', 'finance123')}"
    f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
    f":{os.getenv('POSTGRES_PORT', '5432')}"
    f"/{os.getenv('POSTGRES_DB', 'finance_ml')}"
)

_engine = create_engine(DATABASE_URL)


def get_recent_prices(ticker: str, hours: int = 2) -> pd.DataFrame:
    """Get recent OHLCV prices for a ticker."""
    query = text("""
        SELECT timestamp, open, high, low, close, volume
        FROM raw_prices
        WHERE ticker = :ticker
          AND timestamp >= NOW() - INTERVAL ':hours hours'
        ORDER BY timestamp
    """.replace(":hours", str(int(hours))))
    with _engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    return df


def get_all_prices(ticker: str) -> pd.DataFrame:
    """Get all OHLCV prices for a ticker."""
    query = text("""
        SELECT timestamp, open, high, low, close, volume
        FROM raw_prices
        WHERE ticker = :ticker
        ORDER BY timestamp
    """)
    with _engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    return df


def get_recent_features(ticker: str, limit: int = 100) -> pd.DataFrame:
    """Get recent features for a ticker."""
    query = text("""
        SELECT *
        FROM features
        WHERE ticker = :ticker
        ORDER BY timestamp DESC
        LIMIT :limit
    """)
    with _engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "limit": limit})
    return df.sort_values("timestamp")


def get_predictions(ticker: str | None = None, limit: int = 200) -> pd.DataFrame:
    """Get recent predictions, optionally filtered by ticker."""
    if ticker:
        query = text("""
            SELECT *
            FROM predictions
            WHERE ticker = :ticker
            ORDER BY timestamp DESC
            LIMIT :limit
        """)
        params = {"ticker": ticker, "limit": limit}
    else:
        query = text("""
            SELECT *
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT :limit
        """)
        params = {"limit": limit}

    with _engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    return df.sort_values("timestamp") if not df.empty else df


def get_ticker_list() -> list[str]:
    """Get list of tickers that have data."""
    query = text("SELECT DISTINCT ticker FROM raw_prices ORDER BY ticker")
    with _engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result]


def get_price_count(ticker: str) -> int:
    """Get total price rows for a ticker."""
    query = text("SELECT count(*) FROM raw_prices WHERE ticker = :ticker")
    with _engine.connect() as conn:
        result = conn.execute(query, {"ticker": ticker})
        return result.scalar()


def get_feature_count(ticker: str) -> int:
    """Get total feature rows for a ticker."""
    query = text("SELECT count(*) FROM features WHERE ticker = :ticker")
    with _engine.connect() as conn:
        result = conn.execute(query, {"ticker": ticker})
        return result.scalar()
