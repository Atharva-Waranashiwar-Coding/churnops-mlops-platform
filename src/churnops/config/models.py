"""Typed configuration models for the ChurnOps training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProjectConfig:
    """Project-level metadata and path resolution settings."""

    name: str
    root_dir: Path


@dataclass(slots=True)
class DatasetConfig:
    """Dataset access and schema expectations for churn training."""

    raw_data_path: Path
    target_column: str
    positive_class: str
    column_renames: dict[str, str] = field(default_factory=dict)
    id_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    numeric_coercion_columns: list[str] = field(default_factory=list)
    na_values: list[str] = field(default_factory=list)
    infer_remaining_features: bool = False


@dataclass(slots=True)
class SplitConfig:
    """Train, validation, and test split parameters."""

    test_size: float
    validation_size: float
    random_state: int

    def __post_init__(self) -> None:
        """Validate split ratios after deserialization."""

        if not 0 < self.test_size < 1:
            raise ValueError("split.test_size must be between 0 and 1.")
        if not 0 <= self.validation_size < 1:
            raise ValueError("split.validation_size must be between 0 and 1.")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("split.test_size and split.validation_size must sum to less than 1.")


@dataclass(slots=True)
class ModelConfig:
    """Model selection and estimator parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArtifactConfig:
    """Artifact persistence settings for a training run."""

    root_dir: Path
    training_runs_dir: str
    model_directory: str
    model_filename: str
    metrics_directory: str
    metrics_filename: str
    metadata_directory: str
    metadata_filename: str
    validation_report_filename: str
    config_directory: str
    config_snapshot_filename: str


@dataclass(slots=True)
class ModelRegistryConfig:
    """Model registry settings for tracked training runs."""

    enabled: bool = False
    model_name: str | None = None
    comparison_metric: str = "f1"
    comparison_split: str = "validation"
    greater_is_better: bool = True
    alias: str | None = None

    def __post_init__(self) -> None:
        """Validate registry settings after deserialization."""

        if self.enabled and not self.model_name:
            raise ValueError(
                "tracking.model_registry.model_name must be set when model registry is enabled."
            )


@dataclass(slots=True)
class TrackingConfig:
    """Experiment tracking settings for training runs."""

    enabled: bool = False
    experiment_name: str = "churnops-training"
    tracking_uri: str | None = None
    registry_uri: str | None = None
    artifact_location: str | None = None
    run_name_prefix: str | None = None
    model_artifact_path: str = "model"
    local_artifacts_path: str = "local_run"
    tags: dict[str, str] = field(default_factory=dict)
    model_registry: ModelRegistryConfig = field(default_factory=ModelRegistryConfig)

    def __post_init__(self) -> None:
        """Validate tracking settings after deserialization."""

        if self.enabled and not self.tracking_uri:
            raise ValueError("tracking.tracking_uri must be set when tracking is enabled.")
        if self.enabled and not self.artifact_location:
            raise ValueError("tracking.artifact_location must be set when tracking is enabled.")
        if not self.model_artifact_path:
            raise ValueError("tracking.model_artifact_path must not be empty.")
        if not self.local_artifacts_path:
            raise ValueError("tracking.local_artifacts_path must not be empty.")


@dataclass(slots=True)
class InferenceConfig:
    """Inference service settings for model loading and API execution."""

    model_source: str = "local_artifact"
    local_model_path: Path | None = None
    local_run_id: str | None = None
    model_uri: str | None = None
    registered_model_name: str | None = None
    registered_model_alias: str | None = None
    registered_model_version: str | None = None
    prediction_threshold: float = 0.5
    preload_model: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    def __post_init__(self) -> None:
        """Validate inference settings after deserialization."""

        valid_sources = {"local_artifact", "mlflow_model_uri", "mlflow_registry"}
        if self.model_source not in valid_sources:
            raise ValueError(
                "inference.model_source must be one of: "
                + ", ".join(sorted(valid_sources))
                + "."
            )
        if not 0 <= self.prediction_threshold <= 1:
            raise ValueError("inference.prediction_threshold must be between 0 and 1.")
        if not 1 <= self.port <= 65535:
            raise ValueError("inference.port must be between 1 and 65535.")
        if self.local_model_path is not None and self.local_run_id is not None:
            raise ValueError(
                "Configure either inference.local_model_path or inference.local_run_id, not both."
            )
        if self.model_source == "mlflow_model_uri" and not self.model_uri:
            raise ValueError(
                "inference.model_uri must be set when inference.model_source is "
                "'mlflow_model_uri'."
            )
        if self.model_source == "mlflow_registry":
            if not self.registered_model_name:
                raise ValueError(
                    "inference.registered_model_name must be set when inference.model_source "
                    "is 'mlflow_registry'."
                )
            if bool(self.registered_model_alias) == bool(self.registered_model_version):
                raise ValueError(
                    "Configure exactly one of inference.registered_model_alias or "
                    "inference.registered_model_version for 'mlflow_registry' inference."
                )


@dataclass(slots=True)
class DriftRetrainingConfig:
    """Settings for triggering retraining when drift is detected."""

    enabled: bool = False
    backend: str = "disabled"
    airflow_api_url: str | None = None
    dag_id: str | None = None
    username: str | None = None
    password: str | None = None
    cooldown_minutes: int = 240
    request_timeout_seconds: int = 10

    def __post_init__(self) -> None:
        """Validate retraining trigger settings after deserialization."""

        valid_backends = {"disabled", "airflow_api"}
        if self.backend not in valid_backends:
            raise ValueError(
                "drift.retraining.backend must be one of: "
                + ", ".join(sorted(valid_backends))
                + "."
            )
        if self.enabled and self.backend == "airflow_api" and not self.airflow_api_url:
            raise ValueError(
                "drift.retraining.airflow_api_url must be set when the Airflow trigger is enabled."
            )
        if self.cooldown_minutes < 0:
            raise ValueError("drift.retraining.cooldown_minutes must be zero or greater.")
        if self.request_timeout_seconds < 1:
            raise ValueError(
                "drift.retraining.request_timeout_seconds must be at least 1 second."
            )


@dataclass(slots=True)
class DriftConfig:
    """Reference-baseline, drift detection, and retraining trigger settings."""

    enabled: bool = True
    storage_dir: Path = Path("artifacts/monitoring/drift")
    baseline_filename: str = "drift_baseline.json"
    window_size: int = 200
    min_samples: int = 100
    numeric_bin_count: int = 10
    categorical_top_k: int = 10
    psi_warning_threshold: float = 0.1
    psi_drift_threshold: float = 0.25
    min_drifted_features: int = 2
    retraining: DriftRetrainingConfig = field(default_factory=DriftRetrainingConfig)

    def __post_init__(self) -> None:
        """Validate drift settings after deserialization."""

        if not self.baseline_filename:
            raise ValueError("drift.baseline_filename must not be empty.")
        if self.window_size < 1:
            raise ValueError("drift.window_size must be at least 1.")
        if self.min_samples < 1:
            raise ValueError("drift.min_samples must be at least 1.")
        if self.window_size < self.min_samples:
            raise ValueError(
                "drift.window_size must be greater than or equal to drift.min_samples."
            )
        if self.numeric_bin_count < 1:
            raise ValueError("drift.numeric_bin_count must be at least 1.")
        if self.categorical_top_k < 1:
            raise ValueError("drift.categorical_top_k must be at least 1.")
        if self.psi_warning_threshold < 0:
            raise ValueError("drift.psi_warning_threshold must be zero or greater.")
        if self.psi_drift_threshold < 0:
            raise ValueError("drift.psi_drift_threshold must be zero or greater.")
        if self.psi_warning_threshold > self.psi_drift_threshold:
            raise ValueError(
                "drift.psi_warning_threshold must be less than or equal to "
                "drift.psi_drift_threshold."
            )
        if self.min_drifted_features < 1:
            raise ValueError("drift.min_drifted_features must be at least 1.")


@dataclass(slots=True)
class AirflowConfig:
    """Airflow DAG scheduling and retry settings."""

    dag_id: str = "churnops_training_pipeline"
    schedule: str | None = None
    start_date: datetime = field(
        default_factory=lambda: datetime.fromisoformat("2024-01-01T00:00:00+00:00")
    )
    catchup: bool = False
    max_active_runs: int = 1
    retries: int = 1
    retry_delay_minutes: int = 5
    tags: list[str] = field(default_factory=lambda: ["churnops", "training"])

    def __post_init__(self) -> None:
        """Validate Airflow DAG settings after deserialization."""

        if not self.dag_id:
            raise ValueError("orchestration.airflow.dag_id must not be empty.")
        if self.max_active_runs < 1:
            raise ValueError("orchestration.airflow.max_active_runs must be at least 1.")
        if self.retries < 0:
            raise ValueError("orchestration.airflow.retries must be zero or greater.")
        if self.retry_delay_minutes < 0:
            raise ValueError(
                "orchestration.airflow.retry_delay_minutes must be zero or greater."
            )


@dataclass(slots=True)
class OrchestrationConfig:
    """Runtime settings for orchestrated pipeline execution."""

    workspace_dir: Path
    airflow: AirflowConfig = field(default_factory=AirflowConfig)


@dataclass(slots=True)
class Settings:
    """Top-level application settings for a training run."""

    project: ProjectConfig
    data: DatasetConfig
    split: SplitConfig
    model: ModelConfig
    artifacts: ArtifactConfig
    tracking: TrackingConfig
    inference: InferenceConfig
    drift: DriftConfig
    orchestration: OrchestrationConfig
    config_path: Path
