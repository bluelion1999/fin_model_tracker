"""
Model evaluation utilities — metrics, confusion matrix, classification report.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute classification metrics and return as a dict suitable for MLflow logging."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

    # AUC requires both classes present
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["auc"] = 0.0

    return metrics


def print_evaluation(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, label: str = ""):
    """Print a formatted evaluation summary."""
    metrics = compute_metrics(y_true, y_pred, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    header = f"  {label}  " if label else "  Evaluation  "
    print(f"\n{'='*50}")
    print(f"{header:=^50}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
    print(f"  FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

    return metrics
