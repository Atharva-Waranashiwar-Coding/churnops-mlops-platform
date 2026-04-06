"""Orchestration interfaces for staged training execution."""

from churnops.orchestration.airflow import build_training_dag
from churnops.orchestration.models import TrainingExecutionContext
from churnops.orchestration.stage_store import TrainingStageStore
from churnops.orchestration.training_tasks import (
    bootstrap_orchestrated_run,
    create_training_execution_context,
    run_evaluation_stage,
    run_evaluation_task,
    run_ingestion_stage,
    run_ingestion_task,
    run_preprocessing_stage,
    run_preprocessing_task,
    run_publication_stage,
    run_registration_task,
    run_training_stage,
    run_training_task,
    run_validation_stage,
    run_validation_task,
)

__all__ = [
    "TrainingExecutionContext",
    "TrainingStageStore",
    "bootstrap_orchestrated_run",
    "build_training_dag",
    "create_training_execution_context",
    "run_evaluation_stage",
    "run_evaluation_task",
    "run_ingestion_stage",
    "run_ingestion_task",
    "run_preprocessing_stage",
    "run_preprocessing_task",
    "run_publication_stage",
    "run_registration_task",
    "run_training_stage",
    "run_training_task",
    "run_validation_stage",
    "run_validation_task",
]
