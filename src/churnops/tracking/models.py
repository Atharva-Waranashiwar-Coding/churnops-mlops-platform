"""Tracking models for completed training runs and registry decisions."""

from __future__ import annotations

from dataclasses import dataclass

from churnops.artifacts.persistence import PersistedRun
from churnops.config import Settings
from churnops.data.validation import DatasetValidationReport
from churnops.features.preprocessing import DataSplits, PreparedDataset
from churnops.models.evaluation import EvaluationResult
from churnops.models.training import TrainedModel


@dataclass(slots=True)
class CompletedTrainingRun:
    """All data required to record a completed training run."""

    settings: Settings
    validation_report: DatasetValidationReport
    prepared_dataset: PreparedDataset
    data_splits: DataSplits
    trained_model: TrainedModel
    evaluation_result: EvaluationResult
    persisted_run: PersistedRun


@dataclass(slots=True)
class ModelRegistryResult:
    """Outcome of the model registry decision for a tracked run."""

    attempted: bool
    registered: bool
    status: str
    model_name: str | None = None
    model_version: str | None = None
    metric_name: str | None = None
    metric_split: str | None = None
    candidate_metric: float | None = None
    incumbent_metric: float | None = None
    incumbent_version: str | None = None


@dataclass(slots=True)
class TrackingResult:
    """Experiment tracking details for a completed pipeline run."""

    enabled: bool
    backend: str
    experiment_name: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    tracking_uri: str | None = None
    registry_uri: str | None = None
    artifact_uri: str | None = None
    model_registry: ModelRegistryResult | None = None
