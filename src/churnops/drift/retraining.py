"""Retraining trigger helpers for drift-driven orchestration."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from churnops.config import Settings
from churnops.drift.models import DriftEvaluationResult, RetrainingTriggerResult
from churnops.inference.models import LoadedModel


class RetrainingTrigger(Protocol):
    """Interface for drift-triggered retraining backends."""

    def trigger(
        self,
        evaluation_result: DriftEvaluationResult,
        loaded_model: LoadedModel,
        monitor_key: str,
        event_id: str,
    ) -> RetrainingTriggerResult:
        """Attempt to start a retraining workflow for the current drift event."""


class DisabledRetrainingTrigger:
    """No-op retraining backend."""

    def trigger(
        self,
        evaluation_result: DriftEvaluationResult,
        loaded_model: LoadedModel,
        monitor_key: str,
        event_id: str,
    ) -> RetrainingTriggerResult:
        """Return a disabled trigger result without performing any remote call."""

        del evaluation_result, loaded_model, monitor_key, event_id
        return RetrainingTriggerResult(
            attempted=False,
            triggered=False,
            status="disabled",
            backend="disabled",
        )


class AirflowAPIRetrainingTrigger:
    """Trigger the existing Airflow training DAG via the Airflow stable REST API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def trigger(
        self,
        evaluation_result: DriftEvaluationResult,
        loaded_model: LoadedModel,
        monitor_key: str,
        event_id: str,
    ) -> RetrainingTriggerResult:
        """Submit a DAG run to Airflow using a drift-aware trigger payload."""

        trigger_settings = self._settings.drift.retraining
        dag_id = trigger_settings.dag_id or self._settings.orchestration.airflow.dag_id
        api_base_url = trigger_settings.airflow_api_url
        if not api_base_url or not dag_id:
            return RetrainingTriggerResult(
                attempted=True,
                triggered=False,
                status="misconfigured",
                backend="airflow_api",
                dag_id=dag_id,
                request_url=api_base_url,
                detail="Airflow API URL or DAG ID is not configured.",
            )

        dag_run_id = (
            "drift__" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
        request_url = (
            f"{api_base_url.rstrip('/')}/dags/{quote(dag_id, safe='')}/dagRuns"
        )
        request_body = {
            "dag_run_id": dag_run_id,
            "conf": {
                "event_id": event_id,
                "monitor_key": monitor_key,
                "trigger_source": "drift_monitor",
                "model_source": loaded_model.descriptor.source_type,
                "training_run_id": loaded_model.descriptor.training_run_id,
                "registered_model_name": loaded_model.descriptor.registered_model_name,
                "registered_model_version": loaded_model.descriptor.registered_model_version,
                "drift_status": evaluation_result.status,
                "observed_sample_size": evaluation_result.observed_sample_size,
                "drifted_features": [
                    result.feature_name
                    for result in evaluation_result.feature_results
                    if result.status == "drift"
                ],
                "max_feature_psi": evaluation_result.max_feature_psi,
            },
        }
        request = Request(
            request_url,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **_build_basic_auth_header(
                    trigger_settings.username,
                    trigger_settings.password,
                ),
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=trigger_settings.request_timeout_seconds) as response:
                response_body = response.read().decode("utf-8", errors="replace")
                return RetrainingTriggerResult(
                    attempted=True,
                    triggered=200 <= response.getcode() < 300,
                    status="triggered" if 200 <= response.getcode() < 300 else "http_error",
                    backend="airflow_api",
                    dag_id=dag_id,
                    dag_run_id=dag_run_id,
                    request_url=request_url,
                    response_status_code=response.getcode(),
                    detail=response_body or None,
                )
        except HTTPError as error:
            return RetrainingTriggerResult(
                attempted=True,
                triggered=False,
                status="http_error",
                backend="airflow_api",
                dag_id=dag_id,
                dag_run_id=dag_run_id,
                request_url=request_url,
                response_status_code=error.code,
                detail=error.read().decode("utf-8", errors="replace"),
            )
        except URLError as error:
            return RetrainingTriggerResult(
                attempted=True,
                triggered=False,
                status="request_error",
                backend="airflow_api",
                dag_id=dag_id,
                dag_run_id=dag_run_id,
                request_url=request_url,
                detail=str(error),
            )


def build_retraining_trigger(settings: Settings) -> RetrainingTrigger:
    """Build the configured drift retraining trigger backend."""

    retraining_settings = settings.drift.retraining
    if not retraining_settings.enabled or retraining_settings.backend == "disabled":
        return DisabledRetrainingTrigger()
    if retraining_settings.backend == "airflow_api":
        return AirflowAPIRetrainingTrigger(settings)
    return DisabledRetrainingTrigger()


def _build_basic_auth_header(username: str | None, password: str | None) -> dict[str, str]:
    """Return a basic-auth header when credentials are provided."""

    if username is None or password is None:
        return {}
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}
