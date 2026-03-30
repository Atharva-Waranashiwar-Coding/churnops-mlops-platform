"""Tests for the modular training runner."""

from __future__ import annotations

from pathlib import Path

import yaml

from churnops.config import load_settings
from churnops.pipeline.runner import run_local_training


def test_run_local_training_returns_modular_pipeline_result(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """The runner should execute each modular stage and persist outputs."""

    config_path = _write_training_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    settings = load_settings(config_path)

    result = run_local_training(settings)

    assert result.validation_report.row_count == 24
    assert result.prepared_dataset.feature_spec.numeric_features == [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]
    assert result.data_splits.X_validation is not None
    assert result.evaluation_result.split_sizes == {"train": 12, "test": 6, "validation": 6}
    assert result.persisted_run.model_path.exists()
    assert result.persisted_run.validation_report_path.exists()


def _write_training_config(tmp_path, churn_fixture_path, dataset_config) -> Path:
    """Create a training config for direct runner tests."""

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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path
