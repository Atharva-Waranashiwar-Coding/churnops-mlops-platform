"""Baseline loading helpers for locally persisted and MLflow-backed models."""

from __future__ import annotations

from pathlib import Path

from churnops.config import Settings
from churnops.drift.baseline import load_drift_baseline
from churnops.drift.models import DriftBaseline
from churnops.inference.models import LoadedModel


def load_drift_baseline_for_model(
    settings: Settings,
    loaded_model: LoadedModel,
) -> DriftBaseline | None:
    """Load the drift baseline associated with the currently served model."""

    descriptor = loaded_model.descriptor
    local_baseline_path = _resolve_local_baseline_path(settings, descriptor.local_run_directory)
    if local_baseline_path is not None and local_baseline_path.exists():
        return load_drift_baseline(local_baseline_path)

    if descriptor.training_run_id is None:
        return None

    artifact_path = (
        f"{settings.tracking.local_artifacts_path}/"
        f"{settings.artifacts.metadata_directory}/"
        f"{settings.drift.baseline_filename}"
    )
    try:
        from mlflow import MlflowClient
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "MLflow is required to load a drift baseline for MLflow-backed inference models."
        ) from error

    client = MlflowClient(
        tracking_uri=settings.tracking.tracking_uri,
        registry_uri=settings.tracking.registry_uri,
    )
    downloaded_baseline = client.download_artifacts(
        descriptor.training_run_id,
        artifact_path,
    )
    return load_drift_baseline(downloaded_baseline)


def _resolve_local_baseline_path(
    settings: Settings,
    run_directory: Path | None,
) -> Path | None:
    """Return the local drift baseline path when a standard run directory is known."""

    if run_directory is None:
        return None
    return run_directory / settings.artifacts.metadata_directory / settings.drift.baseline_filename
