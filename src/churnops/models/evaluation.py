"""Evaluation helpers for churn classification models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float | int | None]:
    """Compute standard binary classification metrics for a trained model."""

    predictions = model.predict(features)
    probabilities = _extract_positive_class_scores(model, features)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()

    metrics: dict[str, float | int | None] = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": _safe_metric(roc_auc_score, target, probabilities),
        "average_precision": _safe_metric(average_precision_score, target, probabilities),
        "support": int(target.shape[0]),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    return metrics


def _extract_positive_class_scores(model: Any, features: pd.DataFrame) -> pd.Series | None:
    """Return positive-class probabilities when the estimator exposes them."""

    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(features)[:, 1])
    return None


def _safe_metric(
    metric_function: Any,
    target: pd.Series,
    scores: pd.Series | None,
) -> float | None:
    """Compute a metric when valid target scores are available."""

    if scores is None:
        return None

    try:
        return float(metric_function(target, scores))
    except ValueError:
        return None
