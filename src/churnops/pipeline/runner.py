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
from churnops.tracking import CompletedTrainingRun, TrackingResult, build_training_tracker


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

    tracker = build_training_tracker(settings)
    with tracker.start_run():
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
            trained_model=trained_model,
            evaluation_result=evaluation_result,
            validation_report=validation_report,
        )
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

    return TrainingPipelineResult(
        validation_report=validation_report,
        prepared_dataset=prepared_dataset,
        data_splits=data_splits,
        trained_model=trained_model,
        evaluation_result=evaluation_result,
        persisted_run=persisted_run,
        tracking_result=tracking_result,
    )
