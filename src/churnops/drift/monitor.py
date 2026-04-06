"""Inference-time drift monitoring with event persistence and retraining triggers."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from churnops.config import Settings
from churnops.drift.detector import evaluate_feature_distribution_drift
from churnops.drift.loader import load_drift_baseline_for_model
from churnops.drift.models import DriftBaseline, DriftEvaluationResult, RetrainingTriggerResult
from churnops.drift.retraining import RetrainingTrigger, build_retraining_trigger
from churnops.drift.storage import DriftStore
from churnops.inference.models import LoadedModel

LOGGER = logging.getLogger(__name__)
_MONITOR_KEY_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]+")


class DriftMonitor:
    """Track inference inputs against the training baseline and trigger retraining on drift."""

    def __init__(
        self,
        settings: Settings,
        retraining_trigger: RetrainingTrigger | None = None,
    ) -> None:
        self._settings = settings
        self._retraining_trigger = retraining_trigger or build_retraining_trigger(settings)
        self._baseline_cache: dict[str, DriftBaseline | None] = {}
        self._baseline_error_cache: set[str] = set()

    def observe(self, feature_frame: pd.DataFrame, loaded_model: LoadedModel) -> None:
        """Append inference inputs to the rolling window and evaluate drift when ready."""

        if not self._settings.drift.enabled or feature_frame.empty:
            return

        monitor_key = _build_monitor_key(loaded_model)
        baseline = self._load_baseline(monitor_key, loaded_model)
        if baseline is None:
            return

        drift_store = DriftStore(self._settings, monitor_key)
        observation_window = drift_store.append_observations(
            feature_frame,
            self._settings.drift.window_size,
        )
        state = drift_store.load_state(model_source=loaded_model.descriptor.source_type)
        state.current_window_rows = int(observation_window.shape[0])

        if observation_window.shape[0] < self._settings.drift.min_samples:
            drift_store.save_state(state)
            return

        evaluation_result = evaluate_feature_distribution_drift(
            observation_window,
            baseline,
            self._settings.drift,
        )
        previous_status = state.previous_status
        event_id = _build_event_id(evaluation_result.status)
        trigger_result = self._maybe_trigger_retraining(
            state=state,
            evaluation_result=evaluation_result,
            loaded_model=loaded_model,
            monitor_key=monitor_key,
            event_id=event_id,
        )

        if _should_write_event(previous_status, evaluation_result.status, trigger_result):
            event_path = drift_store.write_event(
                event_id=event_id,
                payload=_build_event_payload(
                    event_id=event_id,
                    monitor_key=monitor_key,
                    evaluation_result=evaluation_result,
                    loaded_model=loaded_model,
                    trigger_result=trigger_result,
                ),
            )
            state.last_event_id = event_id
            _log_event(event_path, evaluation_result, trigger_result)

        state.previous_status = evaluation_result.status
        state.last_evaluated_at_utc = evaluation_result.evaluated_at_utc
        drift_store.save_state(state)

    def _load_baseline(
        self,
        monitor_key: str,
        loaded_model: LoadedModel,
    ) -> DriftBaseline | None:
        """Load and cache the baseline for the currently served model."""

        if monitor_key in self._baseline_cache:
            return self._baseline_cache[monitor_key]

        try:
            baseline = load_drift_baseline_for_model(self._settings, loaded_model)
        except Exception:
            if monitor_key not in self._baseline_error_cache:
                LOGGER.exception("Unable to load drift baseline for monitor '%s'.", monitor_key)
                self._baseline_error_cache.add(monitor_key)
            self._baseline_cache[monitor_key] = None
            return None

        if baseline is None and monitor_key not in self._baseline_error_cache:
            LOGGER.warning(
                "Drift monitoring is enabled but no baseline was found for monitor '%s'.",
                monitor_key,
            )
            self._baseline_error_cache.add(monitor_key)

        self._baseline_cache[monitor_key] = baseline
        return baseline

    def _maybe_trigger_retraining(
        self,
        state,
        evaluation_result: DriftEvaluationResult,
        loaded_model: LoadedModel,
        monitor_key: str,
        event_id: str,
    ) -> RetrainingTriggerResult | None:
        """Trigger retraining when drift is detected and the cooldown has expired."""

        if evaluation_result.status != "drift_detected":
            return None
        if not self._settings.drift.retraining.enabled:
            return None
        if not self._cooldown_elapsed(state.last_triggered_at_utc):
            return RetrainingTriggerResult(
                attempted=False,
                triggered=False,
                status="cooldown_active",
                backend=self._settings.drift.retraining.backend,
                dag_id=(
                    self._settings.drift.retraining.dag_id
                    or self._settings.orchestration.airflow.dag_id
                ),
                detail="Retraining cooldown is still active.",
            )

        trigger_result = self._retraining_trigger.trigger(
            evaluation_result=evaluation_result,
            loaded_model=loaded_model,
            monitor_key=monitor_key,
            event_id=event_id,
        )
        if trigger_result.triggered:
            state.last_triggered_at_utc = evaluation_result.evaluated_at_utc
            state.last_triggered_event_id = event_id
        return trigger_result

    def _cooldown_elapsed(self, last_triggered_at_utc: str | None) -> bool:
        """Return whether the retraining cooldown has elapsed."""

        if not self._settings.drift.retraining.enabled:
            return False
        if last_triggered_at_utc is None:
            return True
        last_triggered_at = datetime.fromisoformat(last_triggered_at_utc)
        cooldown = timedelta(minutes=self._settings.drift.retraining.cooldown_minutes)
        return datetime.now(timezone.utc) - last_triggered_at >= cooldown


def _build_monitor_key(loaded_model: LoadedModel) -> str:
    """Build a stable, filesystem-safe key for one monitored model."""

    descriptor = loaded_model.descriptor
    if descriptor.training_run_id:
        return _sanitize_key(f"run_{descriptor.training_run_id}")
    if descriptor.registered_model_name and descriptor.registered_model_version:
        return _sanitize_key(
            f"registry_{descriptor.registered_model_name}_v{descriptor.registered_model_version}"
        )
    digest = hashlib.sha1(descriptor.source_uri.encode("utf-8")).hexdigest()[:12]
    return _sanitize_key(f"{descriptor.source_type}_{digest}")


def _sanitize_key(value: str) -> str:
    """Return a filesystem-safe monitor key."""

    sanitized_value = _MONITOR_KEY_SANITIZER.sub("_", value).strip("._")
    return sanitized_value or "drift_monitor"


def _build_event_id(status: str) -> str:
    """Build a monotonic event identifier."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{status}"


def _build_event_payload(
    event_id: str,
    monitor_key: str,
    evaluation_result: DriftEvaluationResult,
    loaded_model: LoadedModel,
    trigger_result: RetrainingTriggerResult | None,
) -> dict[str, object]:
    """Build the persisted event payload for drift transitions and triggers."""

    descriptor = loaded_model.descriptor
    return {
        "event_id": event_id,
        "created_at_utc": evaluation_result.evaluated_at_utc,
        "monitor_key": monitor_key,
        "model": {
            "model_name": descriptor.model_name,
            "source_type": descriptor.source_type,
            "source_uri": descriptor.source_uri,
            "training_run_id": descriptor.training_run_id,
            "registered_model_name": descriptor.registered_model_name,
            "registered_model_version": descriptor.registered_model_version,
        },
        "thresholds": {
            "warning_threshold": evaluation_result.warning_threshold,
            "drift_threshold": evaluation_result.drift_threshold,
            "min_drifted_features": evaluation_result.min_drifted_features,
            "window_size": evaluation_result.window_size,
            "min_samples": evaluation_result.min_samples,
        },
        "evaluation": evaluation_result.to_payload(),
        "retraining_trigger": trigger_result.to_payload() if trigger_result is not None else None,
    }


def _should_write_event(
    previous_status: str | None,
    current_status: str,
    trigger_result: RetrainingTriggerResult | None,
) -> bool:
    """Return whether the current evaluation should produce a persisted event."""

    if trigger_result is not None and trigger_result.attempted:
        return True
    return current_status != previous_status


def _log_event(
    event_path,
    evaluation_result: DriftEvaluationResult,
    trigger_result: RetrainingTriggerResult | None,
) -> None:
    """Emit a structured log line for a persisted drift event."""

    drifted_features = [
        result.feature_name
        for result in evaluation_result.feature_results
        if result.status == "drift"
    ]
    if evaluation_result.status == "drift_detected":
        LOGGER.warning(
            "Drift detected | event=%s drifted_features=%s max_psi=%.4f trigger=%s",
            event_path.name,
            ",".join(drifted_features) or "none",
            evaluation_result.max_feature_psi,
            trigger_result.status if trigger_result is not None else "not_attempted",
        )
        return
    if evaluation_result.status == "warning":
        LOGGER.warning(
            "Drift warning | event=%s warning_features=%s max_psi=%.4f",
            event_path.name,
            ",".join(
                result.feature_name
                for result in evaluation_result.feature_results
                if result.status in {"warning", "drift"}
            )
            or "none",
            evaluation_result.max_feature_psi,
        )
        return
    LOGGER.info(
        "Drift state recovered to stable | event=%s previous_alert_cleared=true",
        event_path.name,
    )
