"""Airflow DAG builder for orchestrated ChurnOps training runs."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from churnops.config import load_runtime_settings
from churnops.orchestration.training_tasks import (
    bootstrap_orchestrated_run,
    run_evaluation_task,
    run_ingestion_task,
    run_preprocessing_task,
    run_registration_task,
    run_training_task,
    run_validation_task,
)


def build_training_dag(config_path: str | Path | None = None):
    """Build the Airflow DAG that orchestrates the staged training workflow."""

    try:
        from airflow.decorators import dag, task
        from airflow.operators.python import get_current_context
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Airflow DAG construction requires the 'apache-airflow' package. "
            "Install the orchestration extras or use the Airflow container image."
        ) from error

    settings = load_runtime_settings(config_path)
    airflow_settings = settings.orchestration.airflow
    resolved_config_path = str(settings.config_path)
    default_args = {
        "owner": settings.project.name,
        "retries": airflow_settings.retries,
        "retry_delay": timedelta(minutes=airflow_settings.retry_delay_minutes),
    }

    @dag(
        dag_id=airflow_settings.dag_id,
        schedule=airflow_settings.schedule,
        start_date=airflow_settings.start_date,
        catchup=airflow_settings.catchup,
        max_active_runs=airflow_settings.max_active_runs,
        default_args=default_args,
        tags=airflow_settings.tags,
    )
    def churnops_training_pipeline():
        """Orchestrate ingestion, validation, training, evaluation, and registration."""

        @task(task_id="bootstrap_run_context")
        def bootstrap_context() -> dict[str, str | None]:
            current_context = get_current_context()
            dag_run = current_context.get("dag_run")
            logical_date = current_context.get("logical_date")
            return bootstrap_orchestrated_run(
                config_path=resolved_config_path,
                orchestrator="airflow",
                orchestrator_run_id=getattr(dag_run, "run_id", None),
                logical_date_utc=(
                    logical_date.in_timezone("UTC").isoformat()
                    if logical_date is not None
                    else None
                ),
            )

        @task(task_id="ingest_dataset")
        def ingest_dataset(
            context_payload: dict[str, str | None],
        ) -> dict[str, str | None]:
            return run_ingestion_task(context_payload, config_path=resolved_config_path)

        @task(task_id="validate_dataset")
        def validate_dataset(
            context_payload: dict[str, str | None],
        ) -> dict[str, str | None]:
            return run_validation_task(context_payload, config_path=resolved_config_path)

        @task(task_id="preprocess_dataset")
        def preprocess_dataset(
            context_payload: dict[str, str | None],
        ) -> dict[str, str | None]:
            return run_preprocessing_task(context_payload, config_path=resolved_config_path)

        @task(task_id="train_model")
        def train_model(
            context_payload: dict[str, str | None],
        ) -> dict[str, str | None]:
            return run_training_task(context_payload, config_path=resolved_config_path)

        @task(task_id="evaluate_model")
        def evaluate_model(
            context_payload: dict[str, str | None],
        ) -> dict[str, str | None]:
            return run_evaluation_task(context_payload, config_path=resolved_config_path)

        @task(task_id="register_model")
        def register_model(context_payload: dict[str, str | None]) -> dict[str, object]:
            return run_registration_task(context_payload, config_path=resolved_config_path)

        execution_context = bootstrap_context()
        ingested_dataset = ingest_dataset(execution_context)
        validated_dataset = validate_dataset(ingested_dataset)
        preprocessed_dataset = preprocess_dataset(validated_dataset)
        trained_model = train_model(preprocessed_dataset)
        evaluated_model = evaluate_model(trained_model)
        register_model(evaluated_model)

    return churnops_training_pipeline()
