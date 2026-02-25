"""
Model Registry page — Browse registered models, view metrics, promote models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

st.title("Model Registry")

# List registered models
try:
    registered_models = client.search_registered_models()
except Exception as e:
    st.error(f"Could not connect to MLflow: {e}")
    st.stop()

if not registered_models:
    st.info("No models registered yet. Run `python training/train.py` first.")
    st.stop()

for rm in registered_models:
    st.subheader(f"{rm.name}")

    # Get all versions
    versions = client.search_model_versions(f"name='{rm.name}'")
    if not versions:
        st.info("No versions found.")
        continue

    # Get current production alias
    production_version = None
    try:
        alias_info = client.get_model_version_by_alias(rm.name, "production")
        production_version = alias_info.version
    except Exception:
        pass

    rows = []
    for v in versions:
        # Get run metrics
        metrics = {}
        if v.run_id:
            try:
                run = client.get_run(v.run_id)
                metrics = run.data.metrics
            except Exception:
                pass

        is_prod = str(v.version) == str(production_version)
        rows.append({
            "Version": v.version,
            "Status": "Production" if is_prod else "-",
            "Run ID": v.run_id[:12] if v.run_id else "-",
            "AUC": f"{metrics.get('auc', 0):.4f}" if "auc" in metrics else "-",
            "F1": f"{metrics.get('f1', 0):.4f}" if "f1" in metrics else "-",
            "Accuracy": f"{metrics.get('accuracy', 0):.4f}" if "accuracy" in metrics else "-",
            "Created": str(pd.Timestamp(v.creation_timestamp, unit="ms").strftime("%Y-%m-%d %H:%M")),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Promote model version
    st.markdown("**Promote a version to Production:**")
    version_options = [str(v.version) for v in versions]
    col1, col2 = st.columns([2, 1])

    with col1:
        selected = st.selectbox(
            "Select version",
            version_options,
            index=version_options.index(str(production_version)) if production_version and str(production_version) in version_options else 0,
            key=f"promote_{rm.name}",
        )

    with col2:
        if st.button("Set as Production", key=f"btn_{rm.name}"):
            try:
                client.set_registered_model_alias(rm.name, "production", selected)
                st.success(f"Version {selected} set as production!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    st.markdown("---")

# Experiment runs
st.subheader("Recent Experiment Runs")

try:
    experiments = client.search_experiments()
    for exp in experiments:
        if exp.name.startswith("stock-"):
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=10,
            )
            if runs:
                run_rows = []
                for run in runs:
                    run_rows.append({
                        "Run Name": run.data.tags.get("mlflow.runName", "-"),
                        "Model Type": run.data.params.get("model_type", "-"),
                        "AUC": f"{run.data.metrics.get('auc', 0):.4f}",
                        "F1": f"{run.data.metrics.get('f1', 0):.4f}",
                        "Accuracy": f"{run.data.metrics.get('accuracy', 0):.4f}",
                        "Train Samples": run.data.params.get("n_train", "-"),
                        "Started": pd.Timestamp(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
                    })
                st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"Could not fetch experiments: {e}")

# Link to MLflow UI
st.markdown(f"[Open MLflow UI]({MLFLOW_TRACKING_URI})")
