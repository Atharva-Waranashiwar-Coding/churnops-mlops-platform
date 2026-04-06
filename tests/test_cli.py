"""CLI-level reliability tests for training and serving entrypoints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from churnops.api import app as api_app_module
from churnops.pipeline import train as train_module


def test_entrypoint_modules_import_cleanly_in_fresh_python_process() -> None:
    """Fresh Python processes should import the CLI entrypoints without circular imports."""

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    source_path = str(repo_root / "src")
    env["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else f"{source_path}{os.pathsep}{existing_pythonpath}"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import churnops.pipeline.train; import churnops.api.app",
        ],
        capture_output=True,
        check=False,
        cwd=repo_root,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_training_main_returns_zero_for_fixture_dataset(
    dataset_config,
    churn_fixture_path,
    monkeypatch,
    tmp_path,
) -> None:
    """The training CLI should complete successfully for the checked-in fixture dataset."""

    config_path = _write_training_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    monkeypatch.setattr(sys, "argv", ["churnops-train", "--config", str(config_path)])

    result = train_module.main()

    assert result == 0
    assert next((tmp_path / "artifacts" / "training").iterdir()).is_dir()


def test_training_main_returns_one_for_missing_dataset(
    dataset_config,
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    """The training CLI should fail cleanly when the configured dataset is missing."""

    config_path = _write_training_config(
        tmp_path=tmp_path,
        churn_fixture_path=tmp_path / "missing.csv",
        dataset_config=dataset_config,
    )
    monkeypatch.setattr(sys, "argv", ["churnops-train", "--config", str(config_path)])

    result = train_module.main()
    captured = capsys.readouterr()

    assert result == 1
    assert "Place a churn CSV" in captured.out


def test_api_main_returns_one_for_missing_config(monkeypatch, tmp_path, capsys) -> None:
    """The serving CLI should fail cleanly when the config path is invalid."""

    config_path = tmp_path / "missing.yaml"
    monkeypatch.setattr(sys, "argv", ["churnops-serve", "--config", str(config_path)])

    result = api_app_module.main()
    captured = capsys.readouterr()

    assert result == 1
    assert "Provide a valid config path" in captured.out


def test_api_main_uses_centralized_uvicorn_settings(
    dataset_config,
    churn_fixture_path,
    monkeypatch,
    tmp_path,
) -> None:
    """The serving CLI should start Uvicorn with the shared logging and access-log settings."""

    config_path = _write_training_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
        inference_override={"preload_model": False, "port": 8010},
    )
    uvicorn_call: dict[str, object] = {}

    def fake_uvicorn_run(app, **kwargs) -> None:
        uvicorn_call["app"] = app
        uvicorn_call.update(kwargs)

    monkeypatch.setattr(api_app_module.uvicorn, "run", fake_uvicorn_run)
    monkeypatch.setattr(sys, "argv", ["churnops-serve", "--config", str(config_path)])

    result = api_app_module.main()

    assert result == 0
    assert uvicorn_call["host"] == "127.0.0.1"
    assert uvicorn_call["port"] == 8010
    assert uvicorn_call["access_log"] is False
    assert uvicorn_call["log_config"] is None


def _write_training_config(
    tmp_path,
    churn_fixture_path,
    dataset_config,
    inference_override: dict | None = None,
) -> Path:
    """Create a config file suitable for CLI-level training and serving tests."""

    inference_section = {
        "model_source": "local_artifact",
        "prediction_threshold": 0.5,
        "preload_model": True,
        "host": "127.0.0.1",
        "port": 8000,
    }
    if inference_override is not None:
        inference_section.update(inference_override)

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
                "inference": inference_section,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path
