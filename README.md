# Yahoo Finance ML Dashboard

An end-to-end machine learning system that ingests near-real-time stock data from Yahoo Finance, trains predictive models, and surfaces live predictions and model performance on an interactive dashboard.

**Prediction target:** Binary classification — will a stock's price be higher or lower in 15 minutes?

**Tickers:** AAPL, TSLA, SPY

## Architecture

```
yfinance → Poller (60s) → PostgreSQL → Feature Engineering
                                            ↓
MLflow ← Model Training ← Historical Features
  ↓
FastAPI (inference) → Streamlit Dashboard
  ↓
Evidently (drift monitoring)
```

## Tech Stack

| Layer | Tool |
|-------|------|
| Data ingestion | yfinance + APScheduler |
| Feature store / DB | PostgreSQL 16 |
| Model training | scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Model registry | MLflow Model Registry |
| Inference API | FastAPI |
| Dashboard | Streamlit |
| Drift monitoring | Evidently AI |
| Containerization | Docker + docker-compose |

## Project Structure

```
finance_model_tracker/
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── ingestion/
│   ├── poller.py          # Fetches 1-min OHLCV data every 60s
│   └── features.py        # Feature engineering (RSI, MACD, returns, vol, BB)
├── database/
│   ├── models.py          # SQLAlchemy table definitions
│   └── init.sql           # DB schema
├── training/
│   ├── train.py           # Train models, log to MLflow
│   └── evaluate.py        # Evaluation metrics
├── serving/
│   ├── main.py            # FastAPI app
│   └── predictor.py       # Loads model from MLflow, scores features
├── dashboard/
│   ├── app.py             # Streamlit entrypoint
│   └── pages/             # Live, performance, models, drift pages
├── monitoring/
│   └── drift.py           # Scheduled drift detection
└── notebooks/
    └── exploration.ipynb
```

## Features Engineered

From raw 1-minute OHLCV data, computed per ticker:

- **Returns:** 1-bar, 5-bar, 15-bar log returns
- **Volatility:** Rolling std of returns (5, 15 windows)
- **Volume ratio:** Current volume / 20-bar rolling average
- **RSI:** 14-period relative strength index
- **MACD:** 12/26 EMA diff, signal line, histogram
- **Bollinger Band position:** Price position relative to bands (0-1)
- **Time features:** Hour of day, day of week

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose

### Setup

```bash
# Clone the repo
git clone https://github.com/bluelion1999/fin_model_tracker.git
cd fin_model_tracker

# Copy environment file
cp .env.example .env

# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL
docker-compose up -d

# Verify database is running
docker exec finance_postgres pg_isready -U finance -d finance_ml
```

### Run the Data Pipeline

```bash
# Start the ingestion poller (fetches data every 60s during market hours)
python ingestion/poller.py
```

### Database Schema

Three tables in PostgreSQL:

- **raw_prices** — OHLCV candles per ticker/timestamp
- **features** — Engineered features + labels per ticker/timestamp
- **predictions** — Model predictions with actual outcomes (filled retroactively)

