"""Training pipeline orchestration for local execution."""

from __future__ import annotations

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

    raw_dataset = run_ingestion_stage(settings)
    validation_report = run_validation_stage(raw_dataset, settings)
    prepared_dataset, data_splits = run_preprocessing_stage(raw_dataset, settings)
    trained_model = run_training_stage(prepared_dataset, data_splits, settings)
    evaluation_result = run_evaluation_stage(trained_model, data_splits)
    persisted_run, tracking_result = run_publication_stage(
        settings=settings,
        validation_report=validation_report,
        prepared_dataset=prepared_dataset,
        data_splits=data_splits,
        trained_model=trained_model,
        evaluation_result=evaluation_result,
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
