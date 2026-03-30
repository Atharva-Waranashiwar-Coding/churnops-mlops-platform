"""Tests for centralized training configuration handling."""

from __future__ import annotations

from pathlib import Path

import yaml

from churnops.config import apply_runtime_overrides, ensure_runtime_directories, load_settings


def test_load_settings_applies_artifact_directory_defaults(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Config loading should fill in the standard artifact directory layout by default."""

    config_path = _write_training_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )

    settings = load_settings(config_path)

    assert settings.artifacts.model_directory == "model"
    assert settings.artifacts.metrics_directory == "metrics"
    assert settings.artifacts.metadata_directory == "metadata"
    assert settings.artifacts.config_directory == "config"
    assert settings.tracking.enabled is False
    assert settings.tracking.tracking_uri == _sqlite_uri(tmp_path / "artifacts" / "mlflow" / "mlflow.db")
    assert settings.tracking.registry_uri == settings.tracking.tracking_uri
    assert settings.tracking.artifact_location == (tmp_path / "artifacts" / "mlflow" / "artifacts").resolve().as_uri()
    assert settings.inference.model_source == "local_artifact"
    assert settings.inference.prediction_threshold == 0.5
    assert settings.inference.port == 8000


def test_apply_runtime_overrides_resolves_relative_dataset_paths(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Runtime data-path overrides should resolve relative to the configured project root."""

    config_path = _write_training_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    settings = load_settings(config_path)

    overridden = apply_runtime_overrides(settings, data_path=Path("alternate.csv"))

    assert overridden.data.raw_data_path == (tmp_path / "alternate.csv").resolve()
    assert overridden.project.root_dir == settings.project.root_dir


def test_apply_runtime_overrides_reads_environment_variables(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Runtime env vars should override inference and tracking settings for containers."""

    config_path = _write_training_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    settings = load_settings(config_path)

    overridden = apply_runtime_overrides(
        settings,
        env={
            "CHURNOPS_DATA_PATH": "container-data/customer_churn.csv",
            "CHURNOPS_TRACKING_URI": "sqlite:///runtime/mlflow.db",
            "CHURNOPS_TRACKING_ARTIFACT_LOCATION": "runtime/artifacts",
            "CHURNOPS_INFERENCE_MODEL_SOURCE": "mlflow_registry",
            "CHURNOPS_INFERENCE_REGISTERED_MODEL_NAME": "registry-model",
            "CHURNOPS_INFERENCE_REGISTERED_MODEL_ALIAS": "champion",
            "CHURNOPS_INFERENCE_LOCAL_MODEL_PATH": "",
            "CHURNOPS_INFERENCE_PRELOAD_MODEL": "false",
            "CHURNOPS_INFERENCE_HOST": "0.0.0.0",
            "CHURNOPS_INFERENCE_PORT": "8010",
        },
    )

    assert overridden.data.raw_data_path == (
        tmp_path / "container-data" / "customer_churn.csv"
    ).resolve()
    assert overridden.tracking.tracking_uri == _sqlite_uri(tmp_path / "runtime" / "mlflow.db")
    assert overridden.tracking.artifact_location == (
        tmp_path / "runtime" / "artifacts"
    ).resolve().as_uri()
    assert overridden.inference.model_source == "mlflow_registry"
    assert overridden.inference.registered_model_name == "registry-model"
    assert overridden.inference.registered_model_alias == "champion"
    assert overridden.inference.local_model_path is None
    assert overridden.inference.preload_model is False
    assert overridden.inference.host == "0.0.0.0"
    assert overridden.inference.port == 8010


def test_ensure_runtime_directories_creates_tracking_and_artifact_paths(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Runtime bootstrap should create local directories needed by containerized services."""

    config_path = _write_training_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
        tracking={
            "enabled": True,
            "tracking_uri": "sqlite:///runtime/mlflow/mlflow.db",
            "artifact_location": "runtime/mlflow/artifacts",
        },
    )
    settings = load_settings(config_path)

    ensure_runtime_directories(settings)

    assert (tmp_path / "artifacts").exists()
    assert (tmp_path / "artifacts" / "training").exists()
    assert (tmp_path / "runtime" / "mlflow").exists()
    assert (tmp_path / "runtime" / "mlflow" / "artifacts").exists()


def test_load_settings_resolves_tracking_configuration(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Tracking config should resolve local paths into MLflow-compatible URIs."""

    config_path = _write_training_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
        tracking={
            "enabled": True,
            "experiment_name": "churnops-exp",
            "tracking_uri": "sqlite:///tracking/backend/mlflow.db",
            "registry_uri": "sqlite:///tracking/registry/mlflow.db",
            "artifact_location": "tracking/artifacts",
            "run_name_prefix": "baseline",
            "model_artifact_path": "served_model",
            "local_artifacts_path": "reports",
            "tags": {
                "environment": "test",
            },
            "model_registry": {
                "enabled": True,
                "model_name": "churnops-model",
                "comparison_metric": "roc_auc",
                "comparison_split": "test",
                "greater_is_better": True,
            },
        },
        inference={
            "model_source": "mlflow_registry",
            "registered_model_name": "churnops-model",
            "registered_model_alias": "champion",
            "prediction_threshold": 0.65,
            "preload_model": False,
            "host": "127.0.0.1",
            "port": 9000,
        },
    )

    settings = load_settings(config_path)

    assert settings.tracking.enabled is True
    assert settings.tracking.experiment_name == "churnops-exp"
    assert settings.tracking.tracking_uri == _sqlite_uri(
        tmp_path / "tracking" / "backend" / "mlflow.db"
    )
    assert settings.tracking.registry_uri == _sqlite_uri(
        tmp_path / "tracking" / "registry" / "mlflow.db"
    )
    assert settings.tracking.artifact_location == (
        tmp_path / "tracking" / "artifacts"
    ).resolve().as_uri()
    assert settings.tracking.run_name_prefix == "baseline"
    assert settings.tracking.model_artifact_path == "served_model"
    assert settings.tracking.local_artifacts_path == "reports"
    assert settings.tracking.tags == {"environment": "test"}
    assert settings.tracking.model_registry.enabled is True
    assert settings.tracking.model_registry.model_name == "churnops-model"
    assert settings.tracking.model_registry.comparison_metric == "roc_auc"
    assert settings.tracking.model_registry.comparison_split == "test"
    assert settings.inference.model_source == "mlflow_registry"
    assert settings.inference.registered_model_name == "churnops-model"
    assert settings.inference.registered_model_alias == "champion"
    assert settings.inference.prediction_threshold == 0.65
    assert settings.inference.preload_model is False
    assert settings.inference.host == "127.0.0.1"
    assert settings.inference.port == 9000


def _write_training_config(
    tmp_path,
    churn_fixture_path,
    dataset_config,
    tracking: dict | None = None,
    inference: dict | None = None,
) -> Path:
    """Create a minimal training config for configuration-focused tests."""

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
                **({"tracking": tracking} if tracking is not None else {}),
                **({"inference": inference} if inference is not None else {}),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _sqlite_uri(path: Path) -> str:
    """Return an absolute SQLite URI for a local MLflow metadata database."""

    return f"sqlite:///{path.resolve()}"
