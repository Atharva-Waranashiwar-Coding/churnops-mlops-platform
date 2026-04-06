"""Typed models for drift baselines, evaluations, and retraining triggers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DriftFeatureBaseline:
    """Reference distribution details for a single monitored feature."""

    feature_name: str
    feature_type: str
    sample_size: int
    bucket_labels: list[str]
    expected_distribution: list[float]
    cut_points: list[float] = field(default_factory=list)
    tracked_categories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the feature baseline."""

        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "sample_size": self.sample_size,
            "bucket_labels": self.bucket_labels,
            "expected_distribution": self.expected_distribution,
            "cut_points": self.cut_points,
            "tracked_categories": self.tracked_categories,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DriftFeatureBaseline:
        """Rehydrate a feature baseline from a persisted payload."""

        return cls(
            feature_name=str(payload["feature_name"]),
            feature_type=str(payload["feature_type"]),
            sample_size=int(payload["sample_size"]),
            bucket_labels=[str(label) for label in payload.get("bucket_labels", [])],
            expected_distribution=[
                float(probability) for probability in payload.get("expected_distribution", [])
            ],
            cut_points=[float(value) for value in payload.get("cut_points", [])],
            tracked_categories=[
                str(category) for category in payload.get("tracked_categories", [])
            ],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class DriftBaseline:
    """Reference distributions derived from the training split."""

    created_at_utc: str
    source_split: str
    sample_size: int
    numeric_features: list[str]
    categorical_features: list[str]
    features: dict[str, DriftFeatureBaseline]

    @property
    def feature_count(self) -> int:
        """Return the number of monitored features in the baseline."""

        return len(self.features)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable baseline payload."""

        return {
            "created_at_utc": self.created_at_utc,
            "source_split": self.source_split,
            "sample_size": self.sample_size,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "features": {
                feature_name: feature_baseline.to_payload()
                for feature_name, feature_baseline in self.features.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DriftBaseline:
        """Rehydrate a persisted drift baseline."""

        return cls(
            created_at_utc=str(payload["created_at_utc"]),
            source_split=str(payload["source_split"]),
            sample_size=int(payload["sample_size"]),
            numeric_features=[str(name) for name in payload.get("numeric_features", [])],
            categorical_features=[
                str(name) for name in payload.get("categorical_features", [])
            ],
            features={
                str(feature_name): DriftFeatureBaseline.from_payload(feature_payload)
                for feature_name, feature_payload in dict(payload.get("features", {})).items()
            },
        )


@dataclass(slots=True)
class FeatureDriftResult:
    """Drift score and status for one monitored feature."""

    feature_name: str
    feature_type: str
    psi: float
    status: str
    bucket_labels: list[str]
    baseline_distribution: list[float]
    observed_distribution: list[float]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable feature result."""

        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "psi": self.psi,
            "status": self.status,
            "bucket_labels": self.bucket_labels,
            "baseline_distribution": self.baseline_distribution,
            "observed_distribution": self.observed_distribution,
        }


@dataclass(slots=True)
class DriftEvaluationResult:
    """Aggregate drift decision for one observed inference window."""

    evaluated_at_utc: str
    status: str
    observed_sample_size: int
    min_samples: int
    window_size: int
    warning_threshold: float
    drift_threshold: float
    min_drifted_features: int
    drifted_feature_count: int
    warning_feature_count: int
    max_feature_psi: float
    feature_results: list[FeatureDriftResult]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable evaluation payload."""

        return {
            "evaluated_at_utc": self.evaluated_at_utc,
            "status": self.status,
            "observed_sample_size": self.observed_sample_size,
            "min_samples": self.min_samples,
            "window_size": self.window_size,
            "warning_threshold": self.warning_threshold,
            "drift_threshold": self.drift_threshold,
            "min_drifted_features": self.min_drifted_features,
            "drifted_feature_count": self.drifted_feature_count,
            "warning_feature_count": self.warning_feature_count,
            "max_feature_psi": self.max_feature_psi,
            "feature_results": [result.to_payload() for result in self.feature_results],
        }


@dataclass(slots=True)
class RetrainingTriggerResult:
    """Outcome of a retraining trigger attempt."""

    attempted: bool
    triggered: bool
    status: str
    backend: str
    dag_id: str | None = None
    dag_run_id: str | None = None
    request_url: str | None = None
    response_status_code: int | None = None
    detail: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable trigger payload."""

        return {
            "attempted": self.attempted,
            "triggered": self.triggered,
            "status": self.status,
            "backend": self.backend,
            "dag_id": self.dag_id,
            "dag_run_id": self.dag_run_id,
            "request_url": self.request_url,
            "response_status_code": self.response_status_code,
            "detail": self.detail,
        }


@dataclass(slots=True)
class DriftMonitoringState:
    """Persisted state for one monitored model and inference window."""

    monitor_key: str
    model_source: str
    previous_status: str = "stable"
    current_window_rows: int = 0
    last_evaluated_at_utc: str | None = None
    last_event_id: str | None = None
    last_triggered_at_utc: str | None = None
    last_triggered_event_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable monitoring state payload."""

        return {
            "monitor_key": self.monitor_key,
            "model_source": self.model_source,
            "previous_status": self.previous_status,
            "current_window_rows": self.current_window_rows,
            "last_evaluated_at_utc": self.last_evaluated_at_utc,
            "last_event_id": self.last_event_id,
            "last_triggered_at_utc": self.last_triggered_at_utc,
            "last_triggered_event_id": self.last_triggered_event_id,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DriftMonitoringState:
        """Rehydrate monitoring state from disk."""

        return cls(
            monitor_key=str(payload["monitor_key"]),
            model_source=str(payload["model_source"]),
            previous_status=str(payload.get("previous_status", "stable")),
            current_window_rows=int(payload.get("current_window_rows", 0)),
            last_evaluated_at_utc=(
                str(payload["last_evaluated_at_utc"])
                if payload.get("last_evaluated_at_utc") is not None
                else None
            ),
            last_event_id=(
                str(payload["last_event_id"])
                if payload.get("last_event_id") is not None
                else None
            ),
            last_triggered_at_utc=(
                str(payload["last_triggered_at_utc"])
                if payload.get("last_triggered_at_utc") is not None
                else None
            ),
            last_triggered_event_id=(
                str(payload["last_triggered_event_id"])
                if payload.get("last_triggered_event_id") is not None
                else None
            ),
        )
