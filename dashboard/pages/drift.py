"""
Drift Monitoring page — Shows feature drift status with red/yellow/green indicators.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from monitoring.drift import run_drift_check, FEATURE_COLS, ALERTS_FILE

st.title("Drift Monitoring")

# Show recent alerts
if os.path.exists(ALERTS_FILE):
    with open(ALERTS_FILE) as f:
        try:
            alerts = json.load(f)
        except json.JSONDecodeError:
            alerts = []
    if alerts:
        recent = alerts[-5:]  # last 5 alerts
        for alert in reversed(recent):
            st.error(alert["message"])
        with st.expander(f"View all alerts ({len(alerts)} total)"):
            st.json(alerts[-20:])

st.caption(
    "Compares the distribution of recent features against a reference window. "
    "Uses KS test and Population Stability Index (PSI)."
)

col1, col2 = st.columns([1, 3])
with col1:
    ref_size = st.number_input("Reference window", value=200, min_value=50, max_value=2000, step=50)
    cur_size = st.number_input("Current window", value=50, min_value=10, max_value=500, step=10)
    run_check = st.button("Run Drift Check", type="primary")

if run_check:
    with st.spinner("Running drift analysis..."):
        results = run_drift_check(reference_size=ref_size, current_size=cur_size)

    if not results:
        st.warning("No data available for drift analysis. Ensure features have been computed.")
        st.stop()

    for ticker, data in results.items():
        status = data["overall_status"]
        status_icon = {"green": ":material/check_circle:", "yellow": ":material/warning:", "red": ":material/error:"}
        status_color = {"green": "green", "yellow": "orange", "red": "red"}

        st.subheader(f"{status_icon.get(status, '')} {ticker} — {status.upper()}")
        st.caption(f"Reference: {data['reference_rows']} rows | Current: {data['current_rows']} rows")

        # Feature drift table
        rows = []
        for feature, fdata in data["features"].items():
            rows.append({
                "Feature": feature,
                "Status": fdata["status"].upper(),
                "KS Statistic": fdata["ks_statistic"],
                "KS p-value": fdata["ks_pvalue"],
                "PSI": fdata["psi"],
                "Ref Mean": fdata["ref_mean"],
                "Cur Mean": fdata["cur_mean"],
                "Ref Std": fdata["ref_std"],
                "Cur Std": fdata["cur_std"],
            })

        df = pd.DataFrame(rows)

        # Color code the status column
        def highlight_status(val):
            colors = {"GREEN": "background-color: #1b5e20; color: white",
                      "YELLOW": "background-color: #e65100; color: white",
                      "RED": "background-color: #b71c1c; color: white"}
            return colors.get(val, "")

        styled = df.style.applymap(highlight_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # KS statistic bar chart
        if rows:
            fig = go.Figure()
            colors_list = [
                "#b71c1c" if r["Status"] == "RED" else "#e65100" if r["Status"] == "YELLOW" else "#1b5e20"
                for r in rows
            ]
            fig.add_trace(go.Bar(
                x=[r["Feature"] for r in rows],
                y=[r["KS Statistic"] for r in rows],
                marker_color=colors_list,
                name="KS Statistic",
            ))
            fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Warning")
            fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Alert")
            fig.update_layout(
                title=f"{ticker} — KS Statistics by Feature",
                template="plotly_dark",
                height=350,
                yaxis_title="KS Statistic",
                margin=dict(l=50, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
else:
    st.info("Click **Run Drift Check** to analyze feature distributions.")
