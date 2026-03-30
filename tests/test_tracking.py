"""Integration tests for MLflow-backed experiment tracking."""

from __future__ import annotations

import pytest
import yaml

from churnops.config import load_settings
from churnops.pipeline.runner import run_local_training


@pytest.fixture()
def mlflow_module():
    """Return the MLflow module when available."""

    return pytest.importorskip("mlflow")


def test_run_local_training_logs_mlflow_run_and_registers_model(
    mlflow_module,
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Tracked runs should log searchable metadata and register the winning model."""

    config_path = _write_tracking_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    settings = load_settings(config_path)

    result = run_local_training(settings)
    client = mlflow_module.MlflowClient(
        tracking_uri=settings.tracking.tracking_uri,
        registry_uri=settings.tracking.registry_uri,
    )

    tracked_run = client.get_run(result.tracking_result.run_id)
    root_artifacts = {
        artifact.path for artifact in client.list_artifacts(result.tracking_result.run_id)
    }
    model_versions = client.search_model_versions(
        f"name = '{settings.tracking.model_registry.model_name}'"
    )

    assert result.tracking_result.enabled is True
    assert result.tracking_result.backend == "mlflow"
    assert result.tracking_result.experiment_name == settings.tracking.experiment_name
    assert result.tracking_result.model_registry is not None
    assert result.tracking_result.model_registry.registered is True
    assert tracked_run.data.params["model.name"] == "logistic_regression"
    assert tracked_run.data.params["data.row_count"] == "24"
    assert tracked_run.info.artifact_uri.startswith(settings.tracking.artifact_location)
    assert tracked_run.data.metrics["test_accuracy"] == pytest.approx(
        result.evaluation_result.metrics["test"]["accuracy"]
    )
    assert tracked_run.data.metrics["validation_f1"] == pytest.approx(
        result.evaluation_result.metrics["validation"]["f1"]
    )
    assert settings.tracking.local_artifacts_path in root_artifacts
    assert len(model_versions) == 1
    assert model_versions[0].run_id == result.tracking_result.run_id
    assert model_versions[0].tags["churnops.registry.metric_name"] == "f1"
    assert model_versions[0].tags["churnops.registry.metric_split"] == "validation"


def test_run_local_training_skips_registration_when_metric_does_not_improve(
    mlflow_module,
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """The registry flow should not create a new version when the incumbent is still best."""

    config_path = _write_tracking_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    settings = load_settings(config_path)
    first_result = run_local_training(settings)
    second_result = run_local_training(settings)
    client = mlflow_module.MlflowClient(
        tracking_uri=settings.tracking.tracking_uri,
        registry_uri=settings.tracking.registry_uri,
    )

    model_versions = client.search_model_versions(
        f"name = '{settings.tracking.model_registry.model_name}'"
    )

    assert first_result.tracking_result.model_registry is not None
    assert first_result.tracking_result.model_registry.registered is True
    assert second_result.tracking_result.model_registry is not None
    assert second_result.tracking_result.model_registry.registered is False
    assert second_result.tracking_result.model_registry.status == "not_best"
    assert len(model_versions) == 1


def _write_tracking_config(tmp_path, churn_fixture_path, dataset_config):
    """Create a training config with MLflow tracking enabled."""

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
                    "enabled": True,
                    "experiment_name": "churnops-test-exp",
                    "tracking_uri": "sqlite:///artifacts/mlflow/mlflow.db",
                    "registry_uri": "sqlite:///artifacts/mlflow/mlflow.db",
                    "artifact_location": "artifacts/mlflow/artifacts",
                    "run_name_prefix": "test-run",
                    "model_artifact_path": "model",
                    "local_artifacts_path": "local_run",
                    "tags": {
                        "environment": "test",
                    },
                    "model_registry": {
                        "enabled": True,
                        "model_name": "churnops-test-model",
                        "comparison_metric": "f1",
                        "comparison_split": "validation",
                        "greater_is_better": True,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path
