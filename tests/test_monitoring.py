"""Observability tests for the inference API metrics surface."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from churnops.api.app import create_app
from churnops.pipeline.train import run_training


def test_metrics_endpoint_exposes_request_count_and_latency_metrics(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """The metrics endpoint should expose request counters and latency histograms."""

    config_path = _write_inference_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    run_training(config_path)

    with TestClient(create_app(config_path)) as client:
        health_response = client.get("/health")
        metrics_response = client.get("/metrics")

    assert health_response.status_code == 200
    assert metrics_response.status_code == 200
    assert metrics_response.headers["content-type"].startswith("text/plain")
    assert _find_sample_value(
        metrics_response.text,
        "churnops_api_http_requests_total",
        {
            "method": "GET",
            "route": "/health",
            "status_code": "200",
        },
    ) == 1.0
    assert _find_sample_value(
        metrics_response.text,
        "churnops_api_http_request_duration_seconds_count",
        {
            "method": "GET",
            "route": "/health",
        },
    ) == 1.0


def test_metrics_endpoint_tracks_prediction_volume_and_output_distribution(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Successful prediction traffic should update ML-facing metrics."""

    config_path = _write_inference_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    run_training(config_path)
    payload = {"instances": _build_prediction_payload(churn_fixture_path, row_count=2)}

    with TestClient(create_app(config_path)) as client:
        prediction_response = client.post("/v1/predictions", json=payload)
        metrics_response = client.get("/metrics")

    assert prediction_response.status_code == 200
    assert _find_sample_value(
        metrics_response.text,
        "churnops_inference_prediction_requests_total",
        {"model_source": "local_artifact"},
    ) == 1.0
    assert _find_sample_value(
        metrics_response.text,
        "churnops_inference_prediction_batch_size_count",
        {"model_source": "local_artifact"},
    ) == 1.0
    assert _find_sample_value(
        metrics_response.text,
        "churnops_inference_prediction_batch_size_sum",
        {"model_source": "local_artifact"},
    ) == 2.0
    assert _find_sample_sum(
        metrics_response.text,
        "churnops_inference_prediction_records_total",
        {"model_source": "local_artifact"},
    ) == 2.0
    assert _find_sample_value(
        metrics_response.text,
        "churnops_inference_churn_probability_count",
        {"model_source": "local_artifact"},
    ) == 2.0


def test_metrics_endpoint_tracks_failures_and_model_load_errors(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Failed prediction requests should increment failure and model-load metrics."""

    config_path = _write_inference_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
        inference_override={
            "local_model_path": "artifacts/training/missing/model/model.joblib",
            "preload_model": False,
        },
    )

    with TestClient(create_app(config_path)) as client:
        prediction_response = client.post(
            "/v1/predictions",
            json={"instances": _build_prediction_payload(churn_fixture_path, row_count=1)},
        )
        metrics_response = client.get("/metrics")

    assert prediction_response.status_code == 503
    assert _find_sample_value(
        metrics_response.text,
        "churnops_api_http_request_failures_total",
        {
            "method": "POST",
            "route": "/v1/predictions",
            "status_family": "server_error",
        },
    ) == 1.0
    assert _find_sample_value(
        metrics_response.text,
        "churnops_inference_model_load_total",
        {
            "model_source": "local_artifact",
            "result": "failure",
        },
    ) == 1.0


def _find_sample_value(metrics_text: str, sample_name: str, labels: dict[str, str]) -> float:
    """Return the first sample value matching a metric name and exact labels."""

    for metric_family in text_string_to_metric_families(metrics_text):
        for sample in metric_family.samples:
            if sample.name != sample_name:
                continue
            if all(sample.labels.get(key) == value for key, value in labels.items()):
                return float(sample.value)
    raise AssertionError(f"Metric sample not found: {sample_name} with labels {labels!r}")


def _find_sample_sum(metrics_text: str, sample_name: str, labels: dict[str, str]) -> float:
    """Return the sum of matching sample values for a partially labeled metric."""

    total = 0.0
    for metric_family in text_string_to_metric_families(metrics_text):
        for sample in metric_family.samples:
            if sample.name != sample_name:
                continue
            if all(sample.labels.get(key) == value for key, value in labels.items()):
                total += float(sample.value)
    return total


def _build_prediction_payload(churn_fixture_path: Path, row_count: int) -> list[dict[str, object]]:
    """Build a prediction payload from the churn fixture without target or ID fields."""

    dataframe = (
        pd.read_csv(churn_fixture_path)
        .drop(columns=["customerID", "Churn"])
        .head(row_count)
    )
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
