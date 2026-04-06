"""Drift detection and retraining helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from churnops.drift.models import (
        DriftBaseline,
        DriftEvaluationResult,
        DriftFeatureBaseline,
        DriftMonitoringState,
        FeatureDriftResult,
        RetrainingTriggerResult,
    )
    from churnops.drift.monitor import DriftMonitor

__all__ = [
    "DriftBaseline",
    "DriftEvaluationResult",
    "DriftFeatureBaseline",
    "DriftMonitor",
    "DriftMonitoringState",
    "FeatureDriftResult",
    "RetrainingTriggerResult",
    "build_drift_baseline",
    "evaluate_feature_distribution_drift",
    "load_drift_baseline",
]


def __getattr__(name: str):
    """Resolve public drift helpers lazily to avoid package import cycles."""

    if name in {"build_drift_baseline", "load_drift_baseline"}:
        from churnops.drift.baseline import build_drift_baseline, load_drift_baseline

        exports = {
            "build_drift_baseline": build_drift_baseline,
            "load_drift_baseline": load_drift_baseline,
        }
        return exports[name]

    if name == "evaluate_feature_distribution_drift":
        from churnops.drift.detector import evaluate_feature_distribution_drift

        return evaluate_feature_distribution_drift

    if name in {
        "DriftBaseline",
        "DriftEvaluationResult",
        "DriftFeatureBaseline",
        "DriftMonitoringState",
        "FeatureDriftResult",
        "RetrainingTriggerResult",
    }:
        from churnops.drift.models import (
            DriftBaseline,
            DriftEvaluationResult,
            DriftFeatureBaseline,
            DriftMonitoringState,
            FeatureDriftResult,
            RetrainingTriggerResult,
        )

        exports = {
            "DriftBaseline": DriftBaseline,
            "DriftEvaluationResult": DriftEvaluationResult,
            "DriftFeatureBaseline": DriftFeatureBaseline,
            "DriftMonitoringState": DriftMonitoringState,
            "FeatureDriftResult": FeatureDriftResult,
            "RetrainingTriggerResult": RetrainingTriggerResult,
        }
        return exports[name]

    if name == "DriftMonitor":
        from churnops.drift.monitor import DriftMonitor

        return DriftMonitor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
