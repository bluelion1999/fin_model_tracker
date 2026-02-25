-- Raw price data
CREATE TABLE IF NOT EXISTS raw_prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    UNIQUE(ticker, timestamp)
);

-- Engineered features + label
CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    return_1 FLOAT,
    return_5 FLOAT,
    return_15 FLOAT,
    vol_5 FLOAT,
    vol_15 FLOAT,
    volume_ratio FLOAT,
    rsi_14 FLOAT,
    macd FLOAT,
    macd_signal FLOAT,
    macd_hist FLOAT,
    bb_position FLOAT,
    hour_of_day INT,
    day_of_week INT,
    label INT,  -- 1 if price higher in 15 min, 0 otherwise. NULL until future bar arrives.
    UNIQUE(ticker, timestamp)
);

-- Stored predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(50),
    predicted_label INT,
    probability FLOAT,
    actual_label INT,  -- filled in retroactively
    UNIQUE(ticker, timestamp)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_raw_prices_ticker_ts ON raw_prices(ticker, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_features_ticker_ts ON features(ticker, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker_ts ON predictions(ticker, timestamp DESC);
