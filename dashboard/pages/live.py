"""
Live Prices page — Candlestick chart with model predictions overlay.
Auto-refreshes every 60 seconds.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard.utils import get_all_prices, get_ticker_list, get_recent_features

st.title("Live Prices & Predictions")

tickers = get_ticker_list()
if not tickers:
    st.warning("No data yet. Start the ingestion poller first.")
    st.stop()

col1, col2 = st.columns([1, 3])
with col1:
    ticker = st.selectbox("Ticker", tickers, index=0)
    auto_refresh = st.toggle("Auto-refresh (60s)", value=True)

if auto_refresh:
    st.empty()
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="live_refresh")

# Fetch data
prices = get_all_prices(ticker)
features = get_recent_features(ticker, limit=50)

if prices.empty:
    st.info(f"No price data for {ticker}.")
    st.stop()

with col2:
    latest = prices.iloc[-1]
    prev = prices.iloc[-2] if len(prices) > 1 else latest
    change = latest["close"] - prev["close"]
    pct = (change / prev["close"]) * 100 if prev["close"] != 0 else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Price", f"${latest['close']:.2f}", f"{change:+.2f} ({pct:+.2f}%)")
    m2.metric("Volume", f"{latest['volume']:,.0f}")
    m3.metric("Data Points", f"{len(prices)}")

# Candlestick chart
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.7, 0.3],
    subplot_titles=[f"{ticker} Price", "Volume"],
)

fig.add_trace(
    go.Candlestick(
        x=prices["timestamp"],
        open=prices["open"],
        high=prices["high"],
        low=prices["low"],
        close=prices["close"],
        name="OHLC",
    ),
    row=1, col=1,
)

# Overlay predictions from features (color by label)
if not features.empty and "label" in features.columns:
    labeled = features.dropna(subset=["label"])
    if not labeled.empty:
        up = labeled[labeled["label"] == 1]
        down = labeled[labeled["label"] == 0]

        # Merge with prices to get close price at those timestamps
        import pandas as pd
        prices_indexed = prices.set_index("timestamp")

        for subset, color, name in [(up, "#00c853", "Up"), (down, "#ff1744", "Down")]:
            matched = subset.merge(
                prices[["timestamp", "close"]], on="timestamp", how="inner"
            )
            if not matched.empty:
                fig.add_trace(
                    go.Scatter(
                        x=matched["timestamp"],
                        y=matched["close"],
                        mode="markers",
                        marker=dict(color=color, size=6, symbol="circle"),
                        name=f"Label: {name}",
                        opacity=0.7,
                    ),
                    row=1, col=1,
                )

# Volume bars
colors = ["#00c853" if c >= o else "#ff1744" for c, o in zip(prices["close"], prices["open"])]
fig.add_trace(
    go.Bar(
        x=prices["timestamp"],
        y=prices["volume"],
        marker_color=colors,
        name="Volume",
        showlegend=False,
    ),
    row=2, col=1,
)

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    margin=dict(l=50, r=20, t=40, b=20),
)
fig.update_xaxes(type="date")

st.plotly_chart(fig, use_container_width=True)

# Recent features table
with st.expander("Recent Features", expanded=False):
    if not features.empty:
        display_cols = [
            "timestamp", "rsi_14", "macd", "macd_hist",
            "bb_position", "return_1", "vol_5", "volume_ratio", "label",
        ]
        existing = [c for c in display_cols if c in features.columns]
        st.dataframe(features[existing].tail(20), use_container_width=True)
    else:
        st.info("No features computed yet.")
