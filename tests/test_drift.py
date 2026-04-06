"""Tests for drift baselines, detection, and retraining triggers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from churnops.config import DriftConfig, DriftRetrainingConfig, load_settings
from churnops.drift import DriftMonitor, build_drift_baseline, evaluate_feature_distribution_drift
from churnops.drift.models import RetrainingTriggerResult
from churnops.features.preprocessing import FeatureSpec
from churnops.inference.loader import load_inference_model
from churnops.pipeline.train import run_training


def test_feature_distribution_drift_detector_flags_shifted_window(tmp_path) -> None:
    """PSI-based drift detection should flag materially shifted feature distributions."""

    baseline_frame = pd.DataFrame(
        {
            "tenure": list(range(100)),
            "Contract": ["Month-to-month"] * 50 + ["Two year"] * 50,
        }
    )
    observed_frame = pd.DataFrame(
        {
            "tenure": [240] * 100,
            "Contract": ["Two year"] * 100,
        }
    )
    drift_config = DriftConfig(
        storage_dir=tmp_path / "drift",
        window_size=100,
        min_samples=50,
        numeric_bin_count=5,
        categorical_top_k=2,
        psi_warning_threshold=0.1,
        psi_drift_threshold=0.2,
        min_drifted_features=1,
        retraining=DriftRetrainingConfig(enabled=False),
    )

    baseline = build_drift_baseline(
        feature_frame=baseline_frame,
        feature_spec=FeatureSpec(
            numeric_features=["tenure"],
            categorical_features=["Contract"],
        ),
        config=drift_config,
    )
    result = evaluate_feature_distribution_drift(observed_frame, baseline, drift_config)

    assert result.status == "drift_detected"
    assert result.drifted_feature_count >= 1
    assert result.max_feature_psi >= drift_config.psi_drift_threshold
    assert result.feature_results[0].feature_name in {"tenure", "Contract"}


def test_drift_monitor_persists_event_and_triggers_retraining(
    dataset_config,
    churn_fixture_path,
    tmp_path,
) -> None:
    """Drift monitoring should persist an event and trigger retraining when thresholds are met."""

    config_path = _write_drift_config(
        tmp_path=tmp_path,
        churn_fixture_path=churn_fixture_path,
        dataset_config=dataset_config,
    )
    pipeline_result = run_training(config_path)
    settings = load_settings(config_path)
    loaded_model = load_inference_model(settings)
    trigger = _FakeRetrainingTrigger()
    monitor = DriftMonitor(settings, retraining_trigger=trigger)

    drifted_feature_frame = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "gender": "Female",
                        "SeniorCitizen": 1,
                        "Partner": "No",
                        "Dependents": "No",
                        "tenure": 240,
                        "PhoneService": "Yes",
                        "MultipleLines": "Yes",
                        "InternetService": "Fiber optic",
                        "OnlineSecurity": "No",
                        "OnlineBackup": "No",
                        "DeviceProtection": "No",
                        "TechSupport": "No",
                        "StreamingTV": "Yes",
                        "StreamingMovies": "Yes",
                        "Contract": "Two year",
                        "PaperlessBilling": "Yes",
                        "PaymentMethod": "Electronic check",
                        "MonthlyCharges": 120.0,
                        "TotalCharges": 2200.0,
                    }
                ]
            )
        ]
        * 10,
        ignore_index=True,
    )

    monitor.observe(drifted_feature_frame, loaded_model)

    event_paths = list(settings.drift.storage_dir.rglob("events/*.json"))
    assert len(event_paths) == 1
    event_payload = json.loads(event_paths[0].read_text(encoding="utf-8"))
    state_payload = json.loads(
        next(settings.drift.storage_dir.rglob("state.json")).read_text(encoding="utf-8")
    )

    assert pipeline_result.persisted_run.drift_baseline_path.exists()
    assert event_payload["evaluation"]["status"] == "drift_detected"
    assert event_payload["retraining_trigger"]["status"] == "triggered"
    assert event_payload["thresholds"]["min_drifted_features"] == 1
    assert trigger.call_count == 1
    assert state_payload["current_window_rows"] == 10
    assert state_payload["last_triggered_event_id"] == event_payload["event_id"]


class _FakeRetrainingTrigger:
    """Capture retraining trigger calls without making outbound HTTP requests."""

    def __init__(self) -> None:
        self.call_count = 0

    def trigger(self, evaluation_result, loaded_model, monitor_key, event_id):
        self.call_count += 1
        return RetrainingTriggerResult(
            attempted=True,
            triggered=True,
            status="triggered",
            backend="airflow_api",
            dag_id="churnops_training_pipeline",
            dag_run_id=f"drift__{event_id}",
            request_url="http://airflow.example/api/v1/dags/churnops_training_pipeline/dagRuns",
            response_status_code=200,
            detail=monitor_key + ":" + loaded_model.descriptor.source_type,
        )


def _write_drift_config(
    tmp_path,
    churn_fixture_path,
    dataset_config,
) -> Path:
    """Create a config file with drift detection enabled for inference tests."""

    config_path = tmp_path / "drift.yaml"
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
                "inference": {
                    "model_source": "local_artifact",
                    "preload_model": False,
                    "prediction_threshold": 0.5,
                    "host": "127.0.0.1",
                    "port": 8000,
                },
                "drift": {
                    "enabled": True,
                    "storage_dir": "artifacts/monitoring/drift",
                    "window_size": 10,
                    "min_samples": 10,
                    "numeric_bin_count": 5,
                    "categorical_top_k": 5,
                    "psi_warning_threshold": 0.02,
                    "psi_drift_threshold": 0.05,
                    "min_drifted_features": 1,
                    "retraining": {
                        "enabled": True,
                        "backend": "airflow_api",
                        "airflow_api_url": "http://airflow.example/api/v1",
                        "dag_id": "churnops_training_pipeline",
                        "cooldown_minutes": 30,
                        "request_timeout_seconds": 5,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path
