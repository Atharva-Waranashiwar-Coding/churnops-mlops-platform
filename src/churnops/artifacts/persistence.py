"""Artifact persistence for local churn training runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
from sklearn.pipeline import Pipeline

from churnops.config import Settings
from churnops.features.preprocessing import FeatureSpec


@dataclass(slots=True)
class PersistedRun:
    """Filesystem details for a completed persisted training run."""

    run_id: str
    run_directory: Path


def persist_training_run(
    settings: Settings,
    model_pipeline: Pipeline,
    metrics: dict[str, dict[str, float | int | None]],
    split_sizes: dict[str, int],
    feature_spec: FeatureSpec,
    source_row_count: int,
) -> PersistedRun:
    """Persist the trained pipeline, metrics, and run metadata to disk."""

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_directory = (
        settings.artifacts.root_dir / settings.artifacts.training_runs_dir / run_id
    )
    run_directory.mkdir(parents=True, exist_ok=False)

    model_path = run_directory / settings.artifacts.model_filename
    metrics_path = run_directory / settings.artifacts.metrics_filename
    metadata_path = run_directory / settings.artifacts.metadata_filename
    config_snapshot_path = run_directory / settings.artifacts.config_snapshot_filename

    joblib.dump(model_pipeline, model_path)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, sort_keys=True)

    metadata = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_name": settings.project.name,
        "config_path": str(settings.config_path),
        "data": {
            "raw_data_path": str(settings.data.raw_data_path),
            "target_column": settings.data.target_column,
            "positive_class": settings.data.positive_class,
            "row_count": source_row_count,
            "numeric_features": feature_spec.numeric_features,
            "categorical_features": feature_spec.categorical_features,
        },
        "split_sizes": split_sizes,
        "model": {
            "name": settings.model.name,
            "params": settings.model.params,
        },
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)

    shutil.copy2(settings.config_path, config_snapshot_path)
    return PersistedRun(run_id=run_id, run_directory=run_directory)
