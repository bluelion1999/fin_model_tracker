"""
Streamlit dashboard entrypoint.

Run with: streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Finance ML Dashboard",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

live_page = st.Page("dashboard/pages/live.py", title="Live Prices", icon=":material/show_chart:")
performance_page = st.Page("dashboard/pages/performance.py", title="Model Performance", icon=":material/analytics:")
models_page = st.Page("dashboard/pages/models.py", title="Model Registry", icon=":material/model_training:")
drift_page = st.Page("dashboard/pages/drift.py", title="Drift Monitor", icon=":material/monitoring:")

pg = st.navigation([live_page, performance_page, models_page, drift_page])
pg.run()
