"""Artifact persistence for local churn training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib

from churnops.config import Settings
from churnops.data.validation import DatasetValidationReport
from churnops.models.evaluation import EvaluationResult
from churnops.models.training import TrainedModel


@dataclass(slots=True)
class PersistedRun:
    """Filesystem details for a completed persisted training run."""

    run_id: str
    run_directory: Path
    model_path: Path
    metrics_path: Path
    metadata_path: Path
    validation_report_path: Path
    config_snapshot_path: Path


def persist_training_run(
    settings: Settings,
    trained_model: TrainedModel,
    evaluation_result: EvaluationResult,
    validation_report: DatasetValidationReport,
) -> PersistedRun:
    """Persist the trained pipeline, metrics, and run metadata to disk."""

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_directory = settings.artifacts.root_dir / settings.artifacts.training_runs_dir / run_id
    model_path = run_directory / settings.artifacts.model_directory / settings.artifacts.model_filename
    metrics_path = (
        run_directory / settings.artifacts.metrics_directory / settings.artifacts.metrics_filename
    )
    metadata_path = (
        run_directory / settings.artifacts.metadata_directory / settings.artifacts.metadata_filename
    )
    validation_report_path = (
        run_directory
        / settings.artifacts.metadata_directory
        / settings.artifacts.validation_report_filename
    )
    config_snapshot_path = (
        run_directory
        / settings.artifacts.config_directory
        / settings.artifacts.config_snapshot_filename
    )

    for directory in {
        run_directory,
        model_path.parent,
        metrics_path.parent,
        metadata_path.parent,
        config_snapshot_path.parent,
    }:
        directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(trained_model.model_pipeline, model_path)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(evaluation_result.metrics, metrics_file, indent=2, sort_keys=True)

    with validation_report_path.open("w", encoding="utf-8") as validation_file:
        json.dump(asdict(validation_report), validation_file, indent=2, sort_keys=True)

    metadata = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_name": settings.project.name,
        "entrypoint": "local_training",
        "config_path": str(settings.config_path),
        "artifacts": {
            "model_path": str(model_path.relative_to(run_directory)),
            "metrics_path": str(metrics_path.relative_to(run_directory)),
            "metadata_path": str(metadata_path.relative_to(run_directory)),
            "validation_report_path": str(validation_report_path.relative_to(run_directory)),
            "config_snapshot_path": str(config_snapshot_path.relative_to(run_directory)),
        },
        "data": {
            "raw_data_path": str(settings.data.raw_data_path),
            "target_column": settings.data.target_column,
            "positive_class": settings.data.positive_class,
            "row_count": validation_report.row_count,
            "column_count": validation_report.column_count,
            "validated_columns": validation_report.validated_columns,
            "target_distribution": validation_report.target_distribution,
            "numeric_features": trained_model.feature_spec.numeric_features,
            "categorical_features": trained_model.feature_spec.categorical_features,
        },
        "split_sizes": evaluation_result.split_sizes,
        "model": {
            "name": settings.model.name,
            "params": settings.model.params,
        },
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)

    shutil.copy2(settings.config_path, config_snapshot_path)
    return PersistedRun(
        run_id=run_id,
        run_directory=run_directory,
        model_path=model_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        validation_report_path=validation_report_path,
        config_snapshot_path=config_snapshot_path,
    )
