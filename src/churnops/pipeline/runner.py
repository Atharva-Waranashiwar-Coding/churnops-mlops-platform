"""Training pipeline orchestration for local execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from churnops.artifacts.persistence import PersistedRun
from churnops.config import Settings
from churnops.data.validation import DatasetValidationReport
from churnops.features.preprocessing import DataSplits, PreparedDataset
from churnops.models.evaluation import EvaluationResult
from churnops.models.training import TrainedModel
from churnops.orchestration import (
    run_evaluation_stage,
    run_ingestion_stage,
    run_preprocessing_stage,
    run_publication_stage,
    run_training_stage,
    run_validation_stage,
)
from churnops.tracking import TrackingResult

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingPipelineResult:
    """Full result of a local training pipeline execution."""

    validation_report: DatasetValidationReport
    prepared_dataset: PreparedDataset
    data_splits: DataSplits
    trained_model: TrainedModel
    evaluation_result: EvaluationResult
    persisted_run: PersistedRun
    tracking_result: TrackingResult


def run_local_training(settings: Settings) -> TrainingPipelineResult:
    """Execute the modular local training workflow for a resolved settings object."""

    LOGGER.info(
        "Training pipeline started | project=%s data_path=%s tracking_enabled=%s drift_enabled=%s",
        settings.project.name,
        settings.data.raw_data_path,
        settings.tracking.enabled,
        settings.drift.enabled,
    )
    raw_dataset = run_ingestion_stage(settings)
    LOGGER.info(
        "Ingestion completed | rows=%d columns=%d",
        raw_dataset.shape[0],
        raw_dataset.shape[1],
    )
    validation_report = run_validation_stage(raw_dataset, settings)
    LOGGER.info(
        "Validation completed | required_columns=%d target_column=%s",
        len(settings.data.required_columns),
        settings.data.target_column,
    )
    prepared_dataset, data_splits = run_preprocessing_stage(raw_dataset, settings)
    LOGGER.info(
        "Preprocessing completed | train=%d validation=%d test=%d",
        data_splits.y_train.shape[0],
        data_splits.y_validation.shape[0] if data_splits.y_validation is not None else 0,
        data_splits.y_test.shape[0],
    )
    trained_model = run_training_stage(prepared_dataset, data_splits, settings)
    LOGGER.info(
        "Training completed | model=%s numeric_features=%d categorical_features=%d",
        settings.model.name,
        len(trained_model.feature_spec.numeric_features),
        len(trained_model.feature_spec.categorical_features),
    )
    evaluation_result = run_evaluation_stage(trained_model, data_splits)
    LOGGER.info(
        "Evaluation completed | test_f1=%.4f validation_f1=%s",
        evaluation_result.metrics["test"]["f1"],
        (
            f"{evaluation_result.metrics['validation']['f1']:.4f}"
            if evaluation_result.metrics.get("validation") is not None
            else "n/a"
        ),
    )
    persisted_run, tracking_result = run_publication_stage(
        settings=settings,
        validation_report=validation_report,
        prepared_dataset=prepared_dataset,
        data_splits=data_splits,
        trained_model=trained_model,
        evaluation_result=evaluation_result,
    )
    LOGGER.info(
        "Publication completed | run_id=%s artifact_dir=%s tracking_enabled=%s",
        persisted_run.run_id,
        persisted_run.run_directory,
        tracking_result.enabled,
    )

    return TrainingPipelineResult(
        validation_report=validation_report,
        prepared_dataset=prepared_dataset,
        data_splits=data_splits,
        trained_model=trained_model,
        evaluation_result=evaluation_result,
        persisted_run=persisted_run,
        tracking_result=tracking_result,
    )
