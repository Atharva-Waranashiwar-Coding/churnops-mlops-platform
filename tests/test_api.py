"""API tests for the FastAPI inference service."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import yaml

from churnops.api.app import create_app
from churnops.pipeline.train import run_training


def test_health_endpoint_reports_ready_after_model_preload(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Health should report a ready model when the local artifact loads successfully."""

    config_path = _write_inference_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    run_training(config_path)

    with TestClient(create_app(config_path)) as client:
        response = client.get("/health")

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_source"] == "local_artifact"


def test_model_metadata_endpoint_returns_feature_contract(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Metadata should expose the loaded model source and expected feature groups."""

    config_path = _write_inference_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    run_training(config_path)

    with TestClient(create_app(config_path)) as client:
        response = client.get("/v1/model/metadata")

    payload = response.json()
    assert response.status_code == 200
    assert payload["model"]["source_type"] == "local_artifact"
    assert payload["positive_class_label"] == "Yes"
    assert payload["feature_schema"]["numeric_features"] == [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]
    assert payload["feature_schema"]["categorical_features"][0] == "gender"


def test_prediction_endpoint_returns_batch_predictions(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Prediction should return one response object per validated input row."""

    config_path = _write_inference_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    run_training(config_path)
    payload = {"instances": _build_prediction_payload(churn_fixture_path, row_count=2)}

    with TestClient(create_app(config_path)) as client:
        response = client.post("/v1/predictions", json=payload)

    prediction_payload = response.json()
    assert response.status_code == 200
    assert prediction_payload["model"]["source_type"] == "local_artifact"
    assert len(prediction_payload["predictions"]) == 2
    assert prediction_payload["predictions"][0]["predicted_class"] in {0, 1}
    assert 0 <= prediction_payload["predictions"][0]["churn_probability"] <= 1


def test_prediction_endpoint_rejects_invalid_total_charges(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Schema validation should reject invalid TotalCharges values before prediction."""

    config_path = _write_inference_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    run_training(config_path)
    payload = {"instances": _build_prediction_payload(churn_fixture_path, row_count=1)}
    payload["instances"][0]["TotalCharges"] = "not-a-number"

    with TestClient(create_app(config_path)) as client:
        response = client.post("/v1/predictions", json=payload)

    validation_payload = response.json()
    assert response.status_code == 422
    assert validation_payload["detail"] == "Request validation failed."
    assert validation_payload["errors"][0]["loc"][-1] == "TotalCharges"


def test_prediction_endpoint_returns_503_when_model_is_missing(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Prediction should fail cleanly when the configured artifact cannot be loaded."""

    config_path = _write_inference_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
        inference_override={
            "local_model_path": "artifacts/training/missing/model/model.joblib",
        },
    )

    with TestClient(create_app(config_path)) as client:
        health_response = client.get("/health")
        prediction_response = client.post(
            "/v1/predictions",
            json={"instances": _build_prediction_payload(churn_fixture_path, row_count=1)},
        )

    health_payload = health_response.json()
    assert health_response.status_code == 200
    assert health_payload["status"] == "degraded"
    assert health_payload["model_loaded"] is False
    assert prediction_response.status_code == 503


def _build_prediction_payload(churn_fixture_path: Path, row_count: int) -> list[dict[str, object]]:
    """Build a prediction payload from the churn fixture without target or ID fields."""

    dataframe = pd.read_csv(churn_fixture_path).drop(columns=["customerID", "Churn"]).head(row_count)
    return dataframe.to_dict(orient="records")


def _write_inference_config(
    tmp_path,
    churn_fixture_path,
    dataset_config,
    inference_override: dict | None = None,
) -> Path:
    """Create a config file suitable for local training and inference tests."""

    inference_section = {
        "model_source": "local_artifact",
        "prediction_threshold": 0.5,
        "preload_model": True,
        "host": "127.0.0.1",
        "port": 8000,
    }
    if inference_override is not None:
        inference_section.update(inference_override)

    config_path = tmp_path / "inference.yaml"
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
