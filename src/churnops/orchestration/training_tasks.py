"""Reusable stage-oriented orchestration for churn model training."""

from __future__ import annotations

import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from churnops.artifacts.persistence import PersistedRun, persist_training_run
from churnops.config import Settings, load_runtime_settings
from churnops.data.ingestion import read_raw_dataset
from churnops.data.validation import DatasetValidationReport, validate_raw_dataset
from churnops.features.preprocessing import (
    DataSplits,
    PreparedDataset,
    prepare_training_dataset,
    split_dataset,
)
from churnops.models.evaluation import EvaluationResult, evaluate_model_splits
from churnops.models.training import TrainedModel, train_baseline_model
from churnops.orchestration.models import TrainingExecutionContext
from churnops.orchestration.stage_store import TrainingStageStore
from churnops.tracking import CompletedTrainingRun, TrackingResult, build_training_tracker

_RUN_ID_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]+")


def create_training_execution_context(
    settings: Settings,
    orchestrator: str,
    orchestrator_run_id: str | None = None,
    logical_date_utc: str | None = None,
) -> TrainingExecutionContext:
    """Create a stable execution context for staged training orchestration."""

    source_run_id = orchestrator_run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = _RUN_ID_SANITIZER.sub("_", source_run_id).strip("._") or "training_run"
    context = TrainingExecutionContext(
        run_id=run_id,
        workspace_dir=settings.orchestration.workspace_dir / run_id,
        orchestrator=orchestrator,
        orchestrator_run_id=orchestrator_run_id,
        logical_date_utc=logical_date_utc,
    )
    TrainingStageStore(context).initialize()
    return context


def run_ingestion_stage(settings: Settings) -> pd.DataFrame:
    """Execute the dataset ingestion stage."""

    return read_raw_dataset(settings.data)


def run_validation_stage(
    raw_dataset: pd.DataFrame,
    settings: Settings,
) -> DatasetValidationReport:
    """Execute the dataset validation stage."""

    return validate_raw_dataset(raw_dataset, settings.data)


def run_preprocessing_stage(
    raw_dataset: pd.DataFrame,
    settings: Settings,
) -> tuple[PreparedDataset, DataSplits]:
    """Prepare features and supervised splits for training."""

    prepared_dataset = prepare_training_dataset(raw_dataset, settings.data)
    data_splits = split_dataset(
        prepared_dataset.features,
        prepared_dataset.target,
        settings.split,
    )
    return prepared_dataset, data_splits


def run_training_stage(
    prepared_dataset: PreparedDataset,
    data_splits: DataSplits,
    settings: Settings,
) -> TrainedModel:
    """Train the configured churn baseline model."""

    return train_baseline_model(
        train_features=data_splits.X_train,
        train_target=data_splits.y_train,
        feature_spec=prepared_dataset.feature_spec,
        config=settings.model,
    )


def run_evaluation_stage(
    trained_model: TrainedModel,
    data_splits: DataSplits,
) -> EvaluationResult:
    """Evaluate the trained model across configured dataset splits."""

    return evaluate_model_splits(trained_model.model_pipeline, data_splits)


def run_publication_stage(
    settings: Settings,
    validation_report: DatasetValidationReport,
    prepared_dataset: PreparedDataset,
    data_splits: DataSplits,
    trained_model: TrainedModel,
    evaluation_result: EvaluationResult,
) -> tuple[PersistedRun, TrackingResult]:
    """Persist artifacts and publish the completed training run to tracking backends."""

    persisted_run = persist_training_run(
        settings=settings,
        trained_model=trained_model,
        evaluation_result=evaluation_result,
        validation_report=validation_report,
    )
    tracker = build_training_tracker(settings)
    with tracker.start_run():
        tracking_result = tracker.finalize_run(
            CompletedTrainingRun(
                settings=settings,
                validation_report=validation_report,
                prepared_dataset=prepared_dataset,
                data_splits=data_splits,
                trained_model=trained_model,
                evaluation_result=evaluation_result,
                persisted_run=persisted_run,
            )
        )
    return persisted_run, tracking_result


def bootstrap_orchestrated_run(
    config_path: str | Path | None = None,
    orchestrator: str = "airflow",
    orchestrator_run_id: str | None = None,
    logical_date_utc: str | None = None,
) -> dict[str, str | None]:
    """Create and persist orchestration context for a staged training run."""

    settings = load_runtime_settings(config_path)
    return create_training_execution_context(
        settings=settings,
        orchestrator=orchestrator,
        orchestrator_run_id=orchestrator_run_id,
        logical_date_utc=logical_date_utc,
    ).to_payload()


def run_ingestion_task(
    context_payload: dict[str, str | None],
    config_path: str | Path | None = None,
) -> dict[str, str | None]:
    """Materialize ingestion outputs for orchestration systems."""

    settings = load_runtime_settings(config_path)
    context = TrainingExecutionContext.from_payload(context_payload)
    stage_store = TrainingStageStore(context)
    raw_dataset = run_ingestion_stage(settings)
    stage_store.save_dataframe(raw_dataset)
    stage_store.write_json(
        stage_store.stage_dir("ingestion") / "summary.json",
        {
            "columns": raw_dataset.columns.tolist(),
            "column_count": int(raw_dataset.shape[1]),
            "row_count": int(raw_dataset.shape[0]),
        },
    )
    return context.to_payload()


def run_validation_task(
    context_payload: dict[str, str | None],
    config_path: str | Path | None = None,
) -> dict[str, str | None]:
    """Materialize validation outputs for orchestration systems."""

    settings = load_runtime_settings(config_path)
    context = TrainingExecutionContext.from_payload(context_payload)
    stage_store = TrainingStageStore(context)
    validation_report = run_validation_stage(stage_store.load_dataframe(), settings)
    stage_store.save_validation_report(validation_report)
    return context.to_payload()


def run_preprocessing_task(
    context_payload: dict[str, str | None],
    config_path: str | Path | None = None,
) -> dict[str, str | None]:
    """Materialize preprocessing outputs for orchestration systems."""

    settings = load_runtime_settings(config_path)
    context = TrainingExecutionContext.from_payload(context_payload)
    stage_store = TrainingStageStore(context)
    prepared_dataset, data_splits = run_preprocessing_stage(stage_store.load_dataframe(), settings)
    stage_store.save_prepared_dataset(prepared_dataset)
    stage_store.save_data_splits(data_splits)
    stage_store.write_json(
        stage_store.stage_dir("preprocessing") / "summary.json",
        {
            "categorical_features": prepared_dataset.feature_spec.categorical_features,
            "numeric_features": prepared_dataset.feature_spec.numeric_features,
            "split_sizes": {
                "train": int(data_splits.y_train.shape[0]),
                "test": int(data_splits.y_test.shape[0]),
                "validation": (
                    int(data_splits.y_validation.shape[0])
                    if data_splits.y_validation is not None
                    else 0
                ),
            },
        },
    )
    return context.to_payload()


def run_training_task(
    context_payload: dict[str, str | None],
    config_path: str | Path | None = None,
) -> dict[str, str | None]:
    """Materialize training outputs for orchestration systems."""

    settings = load_runtime_settings(config_path)
    context = TrainingExecutionContext.from_payload(context_payload)
    stage_store = TrainingStageStore(context)
    trained_model = run_training_stage(
        prepared_dataset=stage_store.load_prepared_dataset(),
        data_splits=stage_store.load_data_splits(),
        settings=settings,
    )
    stage_store.save_trained_model(trained_model)
    stage_store.write_json(
        stage_store.stage_dir("training") / "summary.json",
        {
            "categorical_features": trained_model.feature_spec.categorical_features,
            "model_name": settings.model.name,
            "numeric_features": trained_model.feature_spec.numeric_features,
        },
    )
    return context.to_payload()


def run_evaluation_task(
    context_payload: dict[str, str | None],
    config_path: str | Path | None = None,
) -> dict[str, str | None]:
    """Materialize evaluation outputs for orchestration systems."""

    context = TrainingExecutionContext.from_payload(context_payload)
    stage_store = TrainingStageStore(context)
    evaluation_result = run_evaluation_stage(
        trained_model=stage_store.load_trained_model(),
        data_splits=stage_store.load_data_splits(),
    )
    stage_store.save_evaluation_result(evaluation_result)
    return context.to_payload()


def run_registration_task(
    context_payload: dict[str, str | None],
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist artifacts and register the completed run using the configured tracker."""

    settings = load_runtime_settings(config_path)
    context = TrainingExecutionContext.from_payload(context_payload)
    stage_store = TrainingStageStore(context)
    persisted_run, tracking_result = run_publication_stage(
        settings=settings,
        validation_report=stage_store.load_validation_report(),
        prepared_dataset=stage_store.load_prepared_dataset(),
        data_splits=stage_store.load_data_splits(),
        trained_model=stage_store.load_trained_model(),
        evaluation_result=stage_store.load_evaluation_result(),
    )
    summary = {
        "context": context.to_payload(),
        "persisted_run": {
            "config_snapshot_path": str(persisted_run.config_snapshot_path),
            "metadata_path": str(persisted_run.metadata_path),
            "metrics_path": str(persisted_run.metrics_path),
            "model_path": str(persisted_run.model_path),
            "run_directory": str(persisted_run.run_directory),
            "run_id": persisted_run.run_id,
            "validation_report_path": str(persisted_run.validation_report_path),
        },
        "tracking_result": asdict(tracking_result),
    }
    stage_store.write_registration_summary(summary)
    return summary
