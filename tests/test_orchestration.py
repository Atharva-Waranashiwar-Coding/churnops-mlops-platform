"""Tests for staged orchestration helpers used by Airflow and local training."""

from __future__ import annotations

from pathlib import Path

import yaml

from churnops.orchestration import (
    TrainingExecutionContext,
    TrainingStageStore,
    bootstrap_orchestrated_run,
    run_evaluation_task,
    run_ingestion_task,
    run_preprocessing_task,
    run_registration_task,
    run_training_task,
    run_validation_task,
)


def test_orchestration_tasks_materialize_stage_outputs_and_publish_model(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """The staged orchestration helpers should persist each task boundary and final model."""

    config_path = _write_training_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )

    context_payload = bootstrap_orchestrated_run(
        config_path=config_path,
        orchestrator="test",
        orchestrator_run_id="manual__2026-03-30T00:00:00+00:00",
        logical_date_utc="2026-03-30T00:00:00+00:00",
    )
    context = TrainingExecutionContext.from_payload(context_payload)
    stage_store = TrainingStageStore(context)

    context_payload = run_ingestion_task(context_payload, config_path=config_path)
    context_payload = run_validation_task(context_payload, config_path=config_path)
    context_payload = run_preprocessing_task(context_payload, config_path=config_path)
    context_payload = run_training_task(context_payload, config_path=config_path)
    context_payload = run_evaluation_task(context_payload, config_path=config_path)
    registration_summary = run_registration_task(context_payload, config_path=config_path)

    assert context.workspace_dir.exists()
    assert stage_store.context_path.exists()
    assert stage_store.raw_dataset_path.exists()
    assert stage_store.validation_report_path.exists()
    assert stage_store.prepared_dataset_path.exists()
    assert stage_store.data_splits_path.exists()
    assert stage_store.trained_model_path.exists()
    assert stage_store.evaluation_result_path.exists()
    assert stage_store.evaluation_metrics_path.exists()
    assert stage_store.registration_summary_path.exists()
    assert Path(registration_summary["persisted_run"]["model_path"]).exists()
    assert registration_summary["tracking_result"]["enabled"] is False
    assert registration_summary["context"]["orchestrator"] == "test"
    assert registration_summary["context"]["run_id"] == "manual__2026-03-30T00_00_00_00_00"


def _write_training_config(tmp_path, churn_fixture_path, dataset_config) -> Path:
    """Create a training config suitable for orchestration-focused tests."""

    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {
                    "name": "churnops-test",
                    "root_dir": str(tmp_path),
                },
                "data": {
                    "raw_data_path": str(churn_fixture_path),
                    "target_column": dataset_config.target_column,
                    "positive_class": dataset_config.positive_class,
                    "column_renames": dataset_config.column_renames,
                    "id_columns": dataset_config.id_columns,
                    "drop_columns": dataset_config.drop_columns,
                    "required_columns": dataset_config.required_columns,
                    "numeric_features": dataset_config.numeric_features,
                    "categorical_features": dataset_config.categorical_features,
                    "numeric_coercion_columns": dataset_config.numeric_coercion_columns,
                    "na_values": dataset_config.na_values,
                },
                "split": {
                    "test_size": 0.25,
                    "validation_size": 0.25,
                    "random_state": 42,
                },
                "model": {
                    "name": "logistic_regression",
                    "params": {
                        "C": 1.0,
                        "class_weight": "balanced",
                        "max_iter": 1000,
                        "solver": "lbfgs",
                    },
                },
                "artifacts": {
                    "root_dir": "artifacts",
                    "training_runs_dir": "training",
                },
                "tracking": {
                    "enabled": False,
                },
                "orchestration": {
                    "workspace_dir": "artifacts/orchestration",
                    "airflow": {
                        "dag_id": "test_training_dag",
                        "schedule": "0 6 * * *",
                        "start_date": "2024-01-01T00:00:00+00:00",
                        "catchup": False,
                        "max_active_runs": 1,
                        "retries": 1,
                        "retry_delay_minutes": 5,
                        "tags": ["test", "training"],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path
