"""
Model Performance page — Rolling accuracy, confusion matrix, precision/recall by ticker.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dashboard.utils import get_predictions, get_ticker_list, get_recent_features

st.title("Model Performance")

tickers = get_ticker_list()
if not tickers:
    st.warning("No data available.")
    st.stop()

# Check if we have any predictions
predictions = get_predictions(limit=500)

if predictions.empty:
    st.info("No predictions stored yet. Predictions are generated when the inference API is called.")
    st.markdown("---")

    # Fall back to showing feature label distribution as a proxy
    st.subheader("Label Distribution from Features")
    st.caption("Since no predictions are stored yet, showing training label distribution.")

    for ticker in tickers:
        features = get_recent_features(ticker, limit=200)
        if features.empty or "label" not in features.columns:
            continue

        labeled = features.dropna(subset=["label"])
        if labeled.empty:
            continue

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**{ticker}**")
            total = len(labeled)
            up = int(labeled["label"].sum())
            down = total - up
            st.metric("Total Labeled", total)
            st.metric("Up %", f"{up/total*100:.1f}%")
            st.metric("Down %", f"{down/total*100:.1f}%")

        with col2:
            fig = px.histogram(
                labeled, x="label", nbins=2, color="label",
                color_discrete_map={0: "#ff1744", 1: "#00c853"},
                labels={"label": "Direction (0=Down, 1=Up)"},
                title=f"{ticker} Label Distribution",
            )
            fig.update_layout(
                template="plotly_dark",
                height=250,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.stop()

# If we have predictions, show full performance metrics
evaluated = predictions.dropna(subset=["actual_label"])

if evaluated.empty:
    st.info("Predictions exist but none have been backfilled with actual labels yet.")
    st.metric("Total Predictions", len(predictions))
    st.stop()

# Rolling accuracy
st.subheader("Rolling Accuracy (last 100 predictions)")

ticker_filter = st.selectbox("Filter by ticker", ["All"] + tickers)

if ticker_filter != "All":
    eval_filtered = evaluated[evaluated["ticker"] == ticker_filter]
else:
    eval_filtered = evaluated

if len(eval_filtered) > 0:
    eval_filtered = eval_filtered.sort_values("timestamp")
    eval_filtered["correct"] = (eval_filtered["predicted_label"] == eval_filtered["actual_label"]).astype(int)
    eval_filtered["rolling_acc"] = eval_filtered["correct"].rolling(min(100, len(eval_filtered)), min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eval_filtered["timestamp"],
        y=eval_filtered["rolling_acc"],
        mode="lines",
        name="Rolling Accuracy",
        line=dict(color="#2196f3", width=2),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Random (50%)")
    fig.update_layout(
        template="plotly_dark",
        height=350,
        yaxis_title="Accuracy",
        yaxis_range=[0, 1],
        margin=dict(l=50, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    from sklearn.metrics import confusion_matrix, classification_report

    y_true = eval_filtered["actual_label"].values
    y_pred = eval_filtered["predicted_label"].values
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Down", "Up"],
        y=["Down", "Up"],
        text_auto=True,
        color_continuous_scale="Blues",
    )
    fig_cm.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Metrics summary
    st.subheader("Metrics Summary")
    accuracy = (y_true == y_pred).mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Total Evaluated", len(eval_filtered))
    col3.metric("Predictions Pending", len(predictions) - len(evaluated))
