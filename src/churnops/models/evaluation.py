"""Evaluation helpers for churn classification models."""

from __future__ import annotations

from dataclasses import dataclass
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

from churnops.features.preprocessing import DataSplits


@dataclass(slots=True)
class EvaluationResult:
    """Metrics and split sizes computed for a training run."""

    metrics: dict[str, dict[str, float | int | None]]
    split_sizes: dict[str, int]


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


def evaluate_model_splits(model: Any, data_splits: DataSplits) -> EvaluationResult:
    """Evaluate a trained model across each available dataset split."""

    split_frames: dict[str, tuple[pd.DataFrame, pd.Series]] = {
        "train": (data_splits.X_train, data_splits.y_train),
        "test": (data_splits.X_test, data_splits.y_test),
    }

    if data_splits.X_validation is not None and data_splits.y_validation is not None:
        split_frames["validation"] = (data_splits.X_validation, data_splits.y_validation)

    metrics = {
        split_name: evaluate_classifier(model, features, target)
        for split_name, (features, target) in split_frames.items()
    }
    split_sizes = {
        split_name: int(target.shape[0])
        for split_name, (_, target) in split_frames.items()
    }

    return EvaluationResult(metrics=metrics, split_sizes=split_sizes)


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
