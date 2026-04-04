"""Runtime configuration overrides for local and containerized execution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from churnops.config.loader import load_settings
from churnops.config.models import Settings

_UNSET = object()


def apply_runtime_overrides(
    settings: Settings,
    data_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Apply CLI-level and environment-level overrides without mutating the config object."""

    environment = os.environ if env is None else env
    overridden_settings = settings

    resolved_data_path = data_path
    if resolved_data_path is None:
        environment_data_path = _read_environment_value(environment, "CHURNOPS_DATA_PATH")
        if environment_data_path is not _UNSET:
            resolved_data_path = environment_data_path or None

    if resolved_data_path is not None:
        overridden_settings = replace(
            overridden_settings,
            data=replace(
                overridden_settings.data,
                raw_data_path=_resolve_path(overridden_settings, resolved_data_path),
            ),
        )

    tracking_updates: dict[str, str] = {}
    tracking_uri = _read_environment_value(environment, "CHURNOPS_TRACKING_URI")
    if tracking_uri is not _UNSET and tracking_uri:
        tracking_updates["tracking_uri"] = _resolve_uri(overridden_settings, tracking_uri)

    registry_uri = _read_environment_value(environment, "CHURNOPS_REGISTRY_URI")
    if registry_uri is not _UNSET and registry_uri:
        tracking_updates["registry_uri"] = _resolve_uri(overridden_settings, registry_uri)

    artifact_location = _read_environment_value(environment, "CHURNOPS_TRACKING_ARTIFACT_LOCATION")
    if artifact_location is not _UNSET and artifact_location:
        tracking_updates["artifact_location"] = _resolve_uri(overridden_settings, artifact_location)

    if tracking_updates:
        overridden_settings = replace(
            overridden_settings,
            tracking=replace(overridden_settings.tracking, **tracking_updates),
        )

    inference_updates: dict[str, object] = {}

    inference_model_source = _read_environment_value(environment, "CHURNOPS_INFERENCE_MODEL_SOURCE")
    if inference_model_source is not _UNSET and inference_model_source:
        inference_updates["model_source"] = inference_model_source

    local_model_path = _read_environment_value(environment, "CHURNOPS_INFERENCE_LOCAL_MODEL_PATH")
    if local_model_path is not _UNSET:
        inference_updates["local_model_path"] = (
            _resolve_path(overridden_settings, local_model_path)
            if local_model_path
            else None
        )

    local_run_id = _read_environment_value(environment, "CHURNOPS_INFERENCE_LOCAL_RUN_ID")
    if local_run_id is not _UNSET:
        inference_updates["local_run_id"] = local_run_id or None

    model_uri = _read_environment_value(environment, "CHURNOPS_INFERENCE_MODEL_URI")
    if model_uri is not _UNSET:
        inference_updates["model_uri"] = model_uri or None

    registered_model_name = _read_environment_value(
        environment,
        "CHURNOPS_INFERENCE_REGISTERED_MODEL_NAME",
    )
    if registered_model_name is not _UNSET:
        inference_updates["registered_model_name"] = registered_model_name or None

    registered_model_alias = _read_environment_value(
        environment,
        "CHURNOPS_INFERENCE_REGISTERED_MODEL_ALIAS",
    )
    if registered_model_alias is not _UNSET:
        inference_updates["registered_model_alias"] = registered_model_alias or None

    registered_model_version = _read_environment_value(
        environment,
        "CHURNOPS_INFERENCE_REGISTERED_MODEL_VERSION",
    )
    if registered_model_version is not _UNSET:
        inference_updates["registered_model_version"] = registered_model_version or None

    prediction_threshold = _read_environment_value(
        environment,
        "CHURNOPS_INFERENCE_PREDICTION_THRESHOLD",
    )
    if prediction_threshold is not _UNSET and prediction_threshold:
        inference_updates["prediction_threshold"] = float(prediction_threshold)

    preload_model = _read_environment_value(environment, "CHURNOPS_INFERENCE_PRELOAD_MODEL")
    if preload_model is not _UNSET and preload_model:
        inference_updates["preload_model"] = _parse_bool(preload_model)

    host = _read_environment_value(environment, "CHURNOPS_INFERENCE_HOST")
    if host is not _UNSET and host:
        inference_updates["host"] = host

    port = _read_environment_value(environment, "CHURNOPS_INFERENCE_PORT")
    if port is not _UNSET and port:
        inference_updates["port"] = int(port)

    if inference_updates:
        overridden_settings = replace(
            overridden_settings,
            inference=replace(overridden_settings.inference, **inference_updates),
        )

    drift_updates: dict[str, object] = {}

    drift_enabled = _read_environment_value(environment, "CHURNOPS_DRIFT_ENABLED")
    if drift_enabled is not _UNSET and drift_enabled:
        drift_updates["enabled"] = _parse_bool(drift_enabled)

    drift_storage_dir = _read_environment_value(environment, "CHURNOPS_DRIFT_STORAGE_DIR")
    if drift_storage_dir is not _UNSET and drift_storage_dir:
        drift_updates["storage_dir"] = _resolve_path(overridden_settings, drift_storage_dir)

    drift_window_size = _read_environment_value(environment, "CHURNOPS_DRIFT_WINDOW_SIZE")
    if drift_window_size is not _UNSET and drift_window_size:
        drift_updates["window_size"] = int(drift_window_size)

    drift_min_samples = _read_environment_value(environment, "CHURNOPS_DRIFT_MIN_SAMPLES")
    if drift_min_samples is not _UNSET and drift_min_samples:
        drift_updates["min_samples"] = int(drift_min_samples)

    drift_numeric_bin_count = _read_environment_value(
        environment,
        "CHURNOPS_DRIFT_NUMERIC_BIN_COUNT",
    )
    if drift_numeric_bin_count is not _UNSET and drift_numeric_bin_count:
        drift_updates["numeric_bin_count"] = int(drift_numeric_bin_count)

    drift_categorical_top_k = _read_environment_value(
        environment,
        "CHURNOPS_DRIFT_CATEGORICAL_TOP_K",
    )
    if drift_categorical_top_k is not _UNSET and drift_categorical_top_k:
        drift_updates["categorical_top_k"] = int(drift_categorical_top_k)

    drift_warning_threshold = _read_environment_value(
        environment,
        "CHURNOPS_DRIFT_PSI_WARNING_THRESHOLD",
    )
    if drift_warning_threshold is not _UNSET and drift_warning_threshold:
        drift_updates["psi_warning_threshold"] = float(drift_warning_threshold)

    drift_threshold = _read_environment_value(environment, "CHURNOPS_DRIFT_PSI_DRIFT_THRESHOLD")
    if drift_threshold is not _UNSET and drift_threshold:
        drift_updates["psi_drift_threshold"] = float(drift_threshold)

    drift_min_drifted_features = _read_environment_value(
        environment,
        "CHURNOPS_DRIFT_MIN_DRIFTED_FEATURES",
    )
    if drift_min_drifted_features is not _UNSET and drift_min_drifted_features:
        drift_updates["min_drifted_features"] = int(drift_min_drifted_features)

    retraining_updates: dict[str, object] = {}

    retraining_enabled = _read_environment_value(
        environment,
        "CHURNOPS_DRIFT_RETRAINING_ENABLED",
    )
    if retraining_enabled is not _UNSET and retraining_enabled:
        retraining_updates["enabled"] = _parse_bool(retraining_enabled)

    retraining_backend = _read_environment_value(
        environment,
        "CHURNOPS_DRIFT_RETRAINING_BACKEND",
    )
    if retraining_backend is not _UNSET and retraining_backend:
        retraining_updates["backend"] = retraining_backend

    airflow_api_url = _read_environment_value(environment, "CHURNOPS_DRIFT_AIRFLOW_API_URL")
    if airflow_api_url is not _UNSET:
        retraining_updates["airflow_api_url"] = airflow_api_url or None

    retraining_dag_id = _read_environment_value(environment, "CHURNOPS_DRIFT_AIRFLOW_DAG_ID")
    if retraining_dag_id is not _UNSET:
        retraining_updates["dag_id"] = retraining_dag_id or None

    retraining_username = _read_environment_value(
        environment,
        "CHURNOPS_DRIFT_AIRFLOW_USERNAME",
    )
    if retraining_username is not _UNSET:
        retraining_updates["username"] = retraining_username or None

    retraining_password = _read_environment_value(
        environment,
        "CHURNOPS_DRIFT_AIRFLOW_PASSWORD",
    )
    if retraining_password is not _UNSET:
        retraining_updates["password"] = retraining_password or None

    retraining_cooldown = _read_environment_value(
        environment,
        "CHURNOPS_DRIFT_RETRAINING_COOLDOWN_MINUTES",
    )
    if retraining_cooldown is not _UNSET and retraining_cooldown:
        retraining_updates["cooldown_minutes"] = int(retraining_cooldown)

    retraining_timeout = _read_environment_value(
        environment,
        "CHURNOPS_DRIFT_RETRAINING_REQUEST_TIMEOUT_SECONDS",
    )
    if retraining_timeout is not _UNSET and retraining_timeout:
        retraining_updates["request_timeout_seconds"] = int(retraining_timeout)

    if retraining_updates:
        drift_updates["retraining"] = replace(
            overridden_settings.drift.retraining,
            **retraining_updates,
        )

    if drift_updates:
        overridden_settings = replace(
            overridden_settings,
            drift=replace(overridden_settings.drift, **drift_updates),
        )

    orchestration_updates: dict[str, object] = {}

    workspace_dir = _read_environment_value(environment, "CHURNOPS_ORCHESTRATION_WORKSPACE_DIR")
    if workspace_dir is not _UNSET and workspace_dir:
        orchestration_updates["workspace_dir"] = _resolve_path(overridden_settings, workspace_dir)

    airflow_updates: dict[str, object] = {}

    dag_id = _read_environment_value(environment, "CHURNOPS_AIRFLOW_DAG_ID")
    if dag_id is not _UNSET and dag_id:
        airflow_updates["dag_id"] = dag_id

    schedule = _read_environment_value(environment, "CHURNOPS_AIRFLOW_SCHEDULE")
    if schedule is not _UNSET:
        airflow_updates["schedule"] = schedule or None

    catchup = _read_environment_value(environment, "CHURNOPS_AIRFLOW_CATCHUP")
    if catchup is not _UNSET and catchup:
        airflow_updates["catchup"] = _parse_bool(catchup)

    max_active_runs = _read_environment_value(environment, "CHURNOPS_AIRFLOW_MAX_ACTIVE_RUNS")
    if max_active_runs is not _UNSET and max_active_runs:
        airflow_updates["max_active_runs"] = int(max_active_runs)

    retries = _read_environment_value(environment, "CHURNOPS_AIRFLOW_RETRIES")
    if retries is not _UNSET and retries:
        airflow_updates["retries"] = int(retries)

    retry_delay_minutes = _read_environment_value(
        environment,
        "CHURNOPS_AIRFLOW_RETRY_DELAY_MINUTES",
    )
    if retry_delay_minutes is not _UNSET and retry_delay_minutes:
        airflow_updates["retry_delay_minutes"] = int(retry_delay_minutes)

    if airflow_updates:
        orchestration_updates["airflow"] = replace(
            overridden_settings.orchestration.airflow,
            **airflow_updates,
        )

    if orchestration_updates:
        overridden_settings = replace(
            overridden_settings,
            orchestration=replace(overridden_settings.orchestration, **orchestration_updates),
        )

    return overridden_settings


def load_runtime_settings(
    config_path: str | Path | None = None,
    data_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Load settings and apply runtime overrides for the current execution environment."""

    resolved_config_path = config_path or get_default_config_path(env)
    settings = load_settings(resolved_config_path)
    settings = apply_runtime_overrides(settings, data_path=data_path, env=env)
    ensure_runtime_directories(settings)
    return settings


def ensure_runtime_directories(settings: Settings) -> None:
    """Create local runtime directories needed by training, tracking, and containers."""

    settings.artifacts.root_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts.root_dir / settings.artifacts.training_runs_dir).mkdir(
        parents=True,
        exist_ok=True,
    )
    settings.data.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    settings.drift.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.orchestration.workspace_dir.mkdir(parents=True, exist_ok=True)

    tracking_store_path = _local_path_from_uri(settings.tracking.tracking_uri)
    if tracking_store_path is not None:
        if settings.tracking.tracking_uri is not None and settings.tracking.tracking_uri.startswith(
            "sqlite:///"
        ):
            tracking_store_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            tracking_store_path.mkdir(parents=True, exist_ok=True)

    tracking_artifact_path = _local_path_from_uri(settings.tracking.artifact_location)
    if tracking_artifact_path is not None:
        tracking_artifact_path.mkdir(parents=True, exist_ok=True)


def get_default_config_path(env: Mapping[str, str] | None = None) -> str:
    """Return the default config path, optionally overridden by environment."""

    environment = os.environ if env is None else env
    return environment.get("CHURNOPS_CONFIG", "configs/base.yaml")


def _read_environment_value(environment: Mapping[str, str], key: str) -> str | object:
    """Return an environment override when present, else a sentinel."""

    if key not in environment:
        return _UNSET
    return environment[key].strip()


def _resolve_path(settings: Settings, value: str | Path) -> Path:
    """Resolve a runtime path relative to the configured project root."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (settings.project.root_dir / candidate).resolve()


def _resolve_uri(settings: Settings, value: str | Path) -> str:
    """Resolve a runtime URI relative to the configured project root when needed."""

    candidate = str(value)
    if candidate.startswith("sqlite:///") and not candidate.startswith("sqlite:////"):
        return f"sqlite:///{_resolve_path(settings, candidate.removeprefix('sqlite:///'))}"
    if "://" in candidate:
        return candidate
    return _resolve_path(settings, candidate).as_uri()


def _local_path_from_uri(value: str | None) -> Path | None:
    """Return a local filesystem path for supported URI formats."""

    if value is None:
        return None
    if value.startswith("sqlite:///"):
        return Path(value.removeprefix("sqlite:///"))
    if value.startswith("file://"):
        return Path(urlparse(value).path)
    if "://" in value:
        return None
    return Path(value)


def _parse_bool(value: str) -> bool:
    """Parse a boolean environment variable."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unable to parse boolean environment override value: {value!r}")
