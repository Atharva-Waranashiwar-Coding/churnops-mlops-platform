"""Training pipeline orchestration for local execution."""

from __future__ import annotations

from dataclasses import dataclass

from churnops.artifacts.persistence import PersistedRun, persist_training_run
from churnops.config import Settings
from churnops.data.ingestion import read_raw_dataset
from churnops.data.validation import DatasetValidationReport, validate_raw_dataset
from churnops.features.preprocessing import DataSplits, PreparedDataset, prepare_training_dataset, split_dataset
from churnops.models.evaluation import EvaluationResult, evaluate_model_splits
from churnops.models.training import TrainedModel, train_baseline_model


@dataclass(slots=True)
class TrainingPipelineResult:
    """Full result of a local training pipeline execution."""

    validation_report: DatasetValidationReport
    prepared_dataset: PreparedDataset
    data_splits: DataSplits
    trained_model: TrainedModel
    evaluation_result: EvaluationResult
    persisted_run: PersistedRun


def run_local_training(settings: Settings) -> TrainingPipelineResult:
    """Execute the modular local training workflow for a resolved settings object."""

    raw_dataset = read_raw_dataset(settings.data)
    validation_report = validate_raw_dataset(raw_dataset, settings.data)
    prepared_dataset = prepare_training_dataset(raw_dataset, settings.data)
    data_splits = split_dataset(
        prepared_dataset.features,
        prepared_dataset.target,
        settings.split,
    )
    trained_model = train_baseline_model(
        train_features=data_splits.X_train,
        train_target=data_splits.y_train,
        feature_spec=prepared_dataset.feature_spec,
        config=settings.model,
    )
    evaluation_result = evaluate_model_splits(trained_model.model_pipeline, data_splits)
    persisted_run = persist_training_run(
        settings=settings,
        model_pipeline=trained_model.model_pipeline,
        metrics=evaluation_result.metrics,
        split_sizes=evaluation_result.split_sizes,
        feature_spec=trained_model.feature_spec,
        source_row_count=validation_report.row_count,
    )
    return TrainingPipelineResult(
        validation_report=validation_report,
        prepared_dataset=prepared_dataset,
        data_splits=data_splits,
        trained_model=trained_model,
        evaluation_result=evaluation_result,
        persisted_run=persisted_run,
    )
