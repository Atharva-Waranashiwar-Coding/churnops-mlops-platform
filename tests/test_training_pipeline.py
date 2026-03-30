"""Integration tests for the end-to-end local training workflow."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import yaml

from churnops.pipeline.train import run_training


def test_run_training_persists_model_metrics_and_metadata(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """The CLI workflow should persist a reusable model and run artifacts."""

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
                    "model_filename": "model.joblib",
                    "metrics_filename": "metrics.json",
                    "metadata_filename": "metadata.json",
                    "config_snapshot_filename": "config.yaml",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    pipeline_result = run_training(config_path)

    model_path = pipeline_result.persisted_run.run_directory / "model.joblib"
    metrics_path = pipeline_result.persisted_run.run_directory / "metrics.json"
    metadata_path = pipeline_result.persisted_run.run_directory / "metadata.json"
    config_snapshot_path = pipeline_result.persisted_run.run_directory / "config.yaml"

    assert model_path.exists()
    assert metrics_path.exists()
    assert metadata_path.exists()
    assert config_snapshot_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert set(metrics.keys()) == {"train", "validation", "test"}
    assert 0.0 <= metrics["test"]["accuracy"] <= 1.0
    assert metadata["project_name"] == "churnops-test"
    assert metadata["split_sizes"] == {"train": 12, "validation": 6, "test": 6}
    assert pipeline_result.evaluation_result.metrics["test"]["f1"] == metrics["test"]["f1"]
    assert pipeline_result.validation_report.row_count == 24

    persisted_model = joblib.load(model_path)
    sample_frame = pd.read_csv(churn_fixture_path).drop(columns=["customerID", "Churn"]).head(3)
    sample_frame["TotalCharges"] = pd.to_numeric(sample_frame["TotalCharges"], errors="coerce")
    predictions = persisted_model.predict(sample_frame)

    assert len(predictions) == 3


def test_run_training_allows_runtime_dataset_path_override(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """The training workflow should allow a CLI-level dataset path override."""

    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {
                    "name": "churnops-test",
                    "root_dir": str(tmp_path),
                },
                "data": {
                    "raw_data_path": "data/raw/missing.csv",
                    "target_column": dataset_config.target_column,
                    "positive_class": dataset_config.positive_class,
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
                    "model_filename": "model.joblib",
                    "metrics_filename": "metrics.json",
                    "metadata_filename": "metadata.json",
                    "config_snapshot_filename": "config.yaml",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    pipeline_result = run_training(config_path, data_path=churn_fixture_path)

    assert pipeline_result.evaluation_result.metrics["test"]["accuracy"] is not None
    assert (pipeline_result.persisted_run.run_directory / "model.joblib").exists()
