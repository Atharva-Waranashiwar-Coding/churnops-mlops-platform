"""Configuration loading helpers for ChurnOps."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from churnops.config.models import (
    AirflowConfig,
    ArtifactConfig,
    DatasetConfig,
    DriftConfig,
    DriftRetrainingConfig,
    InferenceConfig,
    ModelConfig,
    ModelRegistryConfig,
    OrchestrationConfig,
    ProjectConfig,
    Settings,
    SplitConfig,
    TrackingConfig,
)


def load_settings(config_path: str | Path) -> Settings:
    """Load and validate application settings from a YAML file."""

    resolved_config_path = Path(config_path).expanduser().resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as config_file:
        raw_settings = yaml.safe_load(config_file) or {}

    if not isinstance(raw_settings, dict):
        raise ValueError("Configuration file must contain a YAML mapping at the top level.")

    project_section = _as_mapping(raw_settings.get("project", {}), "project")
    data_section = _as_mapping(raw_settings.get("data", {}), "data")
    split_section = _as_mapping(raw_settings.get("split", {}), "split")
    model_section = _as_mapping(raw_settings.get("model", {}), "model")
    artifacts_section = _as_mapping(raw_settings.get("artifacts", {}), "artifacts")
    tracking_section = _as_mapping(raw_settings.get("tracking", {}), "tracking")
    inference_section = _as_mapping(raw_settings.get("inference", {}), "inference")
    drift_section = _as_mapping(raw_settings.get("drift", {}), "drift")
    orchestration_section = _as_mapping(raw_settings.get("orchestration", {}), "orchestration")

    project_root = Path(project_section.get("root_dir", "."))
    if not project_root.is_absolute():
        project_root = (resolved_config_path.parent / project_root).resolve()

    project = ProjectConfig(
        name=str(project_section.get("name", "churnops")),
        root_dir=project_root,
    )
    dataset = DatasetConfig(
        raw_data_path=_resolve_path(project.root_dir, data_section["raw_data_path"]),
        target_column=str(data_section["target_column"]),
        positive_class=str(data_section.get("positive_class", "Yes")),
        column_renames=_as_string_mapping(
            data_section.get("column_renames", {}),
            "data.column_renames",
        ),
        id_columns=_as_string_list(data_section.get("id_columns", []), "data.id_columns"),
        drop_columns=_as_string_list(data_section.get("drop_columns", []), "data.drop_columns"),
        required_columns=_as_string_list(
            data_section.get("required_columns", []),
            "data.required_columns",
        ),
        numeric_features=_as_string_list(
            data_section.get("numeric_features", []),
            "data.numeric_features",
        ),
        categorical_features=_as_string_list(
            data_section.get("categorical_features", []),
            "data.categorical_features",
        ),
        numeric_coercion_columns=_as_string_list(
            data_section.get("numeric_coercion_columns", []),
            "data.numeric_coercion_columns",
        ),
        na_values=_as_string_list(data_section.get("na_values", []), "data.na_values"),
        infer_remaining_features=bool(data_section.get("infer_remaining_features", False)),
    )
    split = SplitConfig(
        test_size=float(split_section.get("test_size", 0.2)),
        validation_size=float(split_section.get("validation_size", 0.1)),
        random_state=int(split_section.get("random_state", 42)),
    )
    model = ModelConfig(
        name=str(model_section.get("name", "logistic_regression")),
        params=_as_mapping(model_section.get("params", {}), "model.params"),
    )
    artifacts = ArtifactConfig(
        root_dir=_resolve_path(project.root_dir, artifacts_section.get("root_dir", "artifacts")),
        training_runs_dir=str(artifacts_section.get("training_runs_dir", "training")),
        model_directory=str(artifacts_section.get("model_directory", "model")),
        model_filename=str(artifacts_section.get("model_filename", "model.joblib")),
        metrics_directory=str(artifacts_section.get("metrics_directory", "metrics")),
        metrics_filename=str(artifacts_section.get("metrics_filename", "metrics.json")),
        metadata_directory=str(artifacts_section.get("metadata_directory", "metadata")),
        metadata_filename=str(artifacts_section.get("metadata_filename", "run.json")),
        validation_report_filename=str(
            artifacts_section.get("validation_report_filename", "validation.json")
        ),
        config_directory=str(artifacts_section.get("config_directory", "config")),
        config_snapshot_filename=str(
            artifacts_section.get("config_snapshot_filename", "training.yaml")
        ),
    )
    model_registry_section = _as_mapping(
        tracking_section.get("model_registry", {}),
        "tracking.model_registry",
    )
    tracking_uri = _resolve_uri(
        project.root_dir,
        tracking_section.get("tracking_uri", "sqlite:///artifacts/mlflow/mlflow.db"),
    )
    registry_uri = _resolve_uri(
        project.root_dir,
        tracking_section.get("registry_uri", tracking_uri),
    )
    artifact_location = _resolve_uri(
        project.root_dir,
        tracking_section.get("artifact_location", "artifacts/mlflow/artifacts"),
    )
    tracking = TrackingConfig(
        enabled=bool(tracking_section.get("enabled", False)),
        experiment_name=str(tracking_section.get("experiment_name", "churnops-training")),
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        artifact_location=artifact_location,
        run_name_prefix=(
            str(tracking_section["run_name_prefix"])
            if tracking_section.get("run_name_prefix") is not None
            else None
        ),
        model_artifact_path=str(tracking_section.get("model_artifact_path", "model")),
        local_artifacts_path=str(tracking_section.get("local_artifacts_path", "local_run")),
        tags=_as_string_mapping(tracking_section.get("tags", {}), "tracking.tags"),
        model_registry=ModelRegistryConfig(
            enabled=bool(model_registry_section.get("enabled", False)),
            model_name=(
                str(model_registry_section["model_name"])
                if model_registry_section.get("model_name") is not None
                else None
            ),
            comparison_metric=str(model_registry_section.get("comparison_metric", "f1")),
            comparison_split=str(model_registry_section.get("comparison_split", "validation")),
            greater_is_better=bool(model_registry_section.get("greater_is_better", True)),
            alias=(
                str(model_registry_section["alias"])
                if model_registry_section.get("alias") is not None
                else None
            ),
        ),
    )
    inference = InferenceConfig(
        model_source=str(inference_section.get("model_source", "local_artifact")),
        local_model_path=(
            _resolve_path(project.root_dir, inference_section["local_model_path"])
            if inference_section.get("local_model_path") is not None
            else None
        ),
        local_run_id=(
            str(inference_section["local_run_id"])
            if inference_section.get("local_run_id") is not None
            else None
        ),
        model_uri=(
            str(inference_section["model_uri"])
            if inference_section.get("model_uri") is not None
            else None
        ),
        registered_model_name=(
            str(inference_section["registered_model_name"])
            if inference_section.get("registered_model_name") is not None
            else (
                str(model_registry_section["model_name"])
                if model_registry_section.get("model_name") is not None
                else None
            )
        ),
        registered_model_alias=(
            str(inference_section["registered_model_alias"])
            if inference_section.get("registered_model_alias") is not None
            else None
        ),
        registered_model_version=(
            str(inference_section["registered_model_version"])
            if inference_section.get("registered_model_version") is not None
            else None
        ),
        prediction_threshold=float(inference_section.get("prediction_threshold", 0.5)),
        preload_model=bool(inference_section.get("preload_model", True)),
        host=str(inference_section.get("host", "0.0.0.0")),
        port=int(inference_section.get("port", 8000)),
    )
    retraining_section = _as_mapping(drift_section.get("retraining", {}), "drift.retraining")
    drift = DriftConfig(
        enabled=bool(drift_section.get("enabled", True)),
        storage_dir=_resolve_path(
            project.root_dir,
            drift_section.get("storage_dir", "artifacts/monitoring/drift"),
        ),
        baseline_filename=str(drift_section.get("baseline_filename", "drift_baseline.json")),
        window_size=int(drift_section.get("window_size", 200)),
        min_samples=int(drift_section.get("min_samples", 100)),
        numeric_bin_count=int(drift_section.get("numeric_bin_count", 10)),
        categorical_top_k=int(drift_section.get("categorical_top_k", 10)),
        psi_warning_threshold=float(drift_section.get("psi_warning_threshold", 0.1)),
        psi_drift_threshold=float(drift_section.get("psi_drift_threshold", 0.25)),
        min_drifted_features=int(drift_section.get("min_drifted_features", 2)),
        retraining=DriftRetrainingConfig(
            enabled=bool(retraining_section.get("enabled", False)),
            backend=str(retraining_section.get("backend", "disabled")),
            airflow_api_url=(
                str(retraining_section["airflow_api_url"])
                if retraining_section.get("airflow_api_url") is not None
                else None
            ),
            dag_id=(
                str(retraining_section["dag_id"])
                if retraining_section.get("dag_id") is not None
                else None
            ),
            username=(
                str(retraining_section["username"])
                if retraining_section.get("username") is not None
                else None
            ),
            password=(
                str(retraining_section["password"])
                if retraining_section.get("password") is not None
                else None
            ),
            cooldown_minutes=int(retraining_section.get("cooldown_minutes", 240)),
            request_timeout_seconds=int(retraining_section.get("request_timeout_seconds", 10)),
        ),
    )
    airflow_section = _as_mapping(orchestration_section.get("airflow", {}), "orchestration.airflow")
    orchestration = OrchestrationConfig(
        workspace_dir=_resolve_path(
            project.root_dir,
            orchestration_section.get("workspace_dir", "artifacts/orchestration"),
        ),
        airflow=AirflowConfig(
            dag_id=str(airflow_section.get("dag_id", "churnops_training_pipeline")),
            schedule=(
                str(airflow_section["schedule"])
                if airflow_section.get("schedule") is not None
                else None
            ),
            start_date=_parse_datetime(
                airflow_section.get("start_date", "2024-01-01T00:00:00+00:00"),
                "orchestration.airflow.start_date",
            ),
            catchup=bool(airflow_section.get("catchup", False)),
            max_active_runs=int(airflow_section.get("max_active_runs", 1)),
            retries=int(airflow_section.get("retries", 1)),
            retry_delay_minutes=int(airflow_section.get("retry_delay_minutes", 5)),
            tags=_as_string_list(
                airflow_section.get("tags", ["churnops", "training"]),
                "orchestration.airflow.tags",
            ),
        ),
    )

    return Settings(
        project=project,
        data=dataset,
        split=split,
        model=model,
        artifacts=artifacts,
        tracking=tracking,
        inference=inference,
        drift=drift,
        orchestration=orchestration,
        config_path=resolved_config_path,
    )


def _resolve_path(base_dir: Path, candidate: str | Path) -> Path:
    """Resolve a potentially relative path against the configured project root."""

    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_uri(base_dir: Path, candidate: str | Path) -> str:
    """Resolve a local path candidate into a file URI when no URI scheme is present."""

    candidate_text = str(candidate)
    if candidate_text.startswith("sqlite:///") and not candidate_text.startswith("sqlite:////"):
        resolved_path = _resolve_path(base_dir, candidate_text.removeprefix("sqlite:///"))
        return f"sqlite:///{resolved_path}"
    if "://" in candidate_text:
        return candidate_text

    return _resolve_path(base_dir, candidate_text).as_uri()


def _as_mapping(value: Any, section_name: str) -> dict[str, Any]:
    """Ensure a configuration section is a mapping."""

    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{section_name}' must be a mapping.")
    return value


def _as_string_list(value: Any, section_name: str) -> list[str]:
    """Ensure a configuration field is a list of strings."""

    if not isinstance(value, list):
        raise ValueError(f"Configuration section '{section_name}' must be a list.")

    string_list = [str(item) for item in value]
    if len(set(string_list)) != len(string_list):
        raise ValueError(f"Configuration section '{section_name}' contains duplicate values.")
    return string_list


def _as_string_mapping(value: Any, section_name: str) -> dict[str, str]:
    """Ensure a configuration field is a string-to-string mapping."""

    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{section_name}' must be a mapping.")

    return {str(key): str(item) for key, item in value.items()}


def _parse_datetime(value: Any, section_name: str) -> datetime:
    """Parse an ISO 8601 datetime configuration value."""

    try:
        return datetime.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(
            f"Configuration section '{section_name}' must be a valid ISO 8601 datetime."
        ) from error
