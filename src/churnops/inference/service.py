"""Inference service layer for loading models and serving predictions."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

import pandas as pd

from churnops.config import Settings
from churnops.drift import DriftMonitor
from churnops.inference.exceptions import ModelLoadError, PredictionError
from churnops.inference.loader import load_inference_model
from churnops.inference.models import LoadedModel, PredictionRecord
from churnops.monitoring.metrics import InferenceMetrics

LOGGER = logging.getLogger(__name__)


class InferenceService:
    """Load and query the configured churn inference model."""

    def __init__(
        self,
        settings: Settings,
        metrics: InferenceMetrics | None = None,
        drift_monitor: DriftMonitor | None = None,
    ) -> None:
        self._settings = settings
        self._metrics = metrics
        self._drift_monitor = drift_monitor or DriftMonitor(settings)
        self._loaded_model: LoadedModel | None = None
        self._last_error: str | None = None
        self._lock = Lock()

    def preload_model(self) -> None:
        """Eagerly load the configured model during application startup."""

        if self._settings.inference.preload_model:
            self.load_model()

    def load_model(self, force_reload: bool = False) -> LoadedModel:
        """Load and cache the configured model."""

        with self._lock:
            if self._loaded_model is not None and not force_reload:
                return self._loaded_model

            try:
                loaded_model = load_inference_model(self._settings)
            except Exception as error:
                self._last_error = str(error)
                if self._metrics is not None:
                    self._metrics.record_model_load(
                        model_source=self._settings.inference.model_source,
                        result="failure",
                    )
                if isinstance(error, ModelLoadError):
                    raise
                raise ModelLoadError(str(error)) from error

            self._loaded_model = loaded_model
            self._last_error = None
            if self._metrics is not None:
                self._metrics.record_model_load(
                    model_source=loaded_model.descriptor.source_type,
                    result="success",
                )
            return loaded_model

    def predict(self, records: list[dict[str, Any]]) -> tuple[LoadedModel, list[PredictionRecord]]:
        """Run batch inference over validated request payloads."""

        loaded_model = self.load_model()
        feature_frame = self._build_feature_frame(records, loaded_model)

        try:
            predictions = loaded_model.predictor.predict(feature_frame)
        except Exception as error:
            raise PredictionError(f"Prediction failed: {error}") from error

        probabilities = self._extract_positive_class_probabilities(
            loaded_model.predictor,
            feature_frame,
        )
        prediction_records = [
            PredictionRecord(
                index=index,
                predicted_class=int(predicted_class),
                predicted_churn=bool(int(predicted_class) == 1),
                churn_probability=(
                    float(probabilities[index]) if probabilities is not None else None
                ),
            )
            for index, predicted_class in enumerate(predictions)
        ]
        if self._metrics is not None:
            self._metrics.record_prediction_batch(
                model_source=loaded_model.descriptor.source_type,
                predictions=prediction_records,
            )
        self._record_drift_observation(feature_frame, loaded_model)
        return loaded_model, prediction_records

    def get_model_metadata(self) -> LoadedModel:
        """Return metadata for the currently configured inference model."""

        return self.load_model()

    def get_health(self) -> dict[str, Any]:
        """Return health information without failing the endpoint."""

        model_loaded = self._loaded_model is not None
        status = "ok" if model_loaded or self._last_error is None else "degraded"
        return {
            "status": status,
            "model_loaded": model_loaded,
            "model_source": self._settings.inference.model_source,
            "last_error": self._last_error,
        }

    def is_ready(self) -> bool:
        """Return whether the service is ready to receive traffic."""

        if self._loaded_model is not None:
            return True
        return not self._settings.inference.preload_model and self._last_error is None

    def _record_drift_observation(
        self,
        feature_frame: pd.DataFrame,
        loaded_model: LoadedModel,
    ) -> None:
        """Pass successful inference inputs into the drift monitor without blocking predictions."""

        try:
            self._drift_monitor.observe(feature_frame, loaded_model)
        except Exception:
            LOGGER.exception("Drift monitoring failed for an inference batch.")

    def _build_feature_frame(
        self,
        records: list[dict[str, Any]],
        loaded_model: LoadedModel,
    ) -> pd.DataFrame:
        """Convert request payloads into the feature frame expected by the model."""

        feature_frame = pd.DataFrame(records)
        expected_features = loaded_model.descriptor.feature_names
        feature_frame = feature_frame.reindex(columns=expected_features)

        for column in self._settings.data.numeric_coercion_columns:
            if column in feature_frame.columns:
                feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")

        return feature_frame

    @staticmethod
    def _extract_positive_class_probabilities(
        predictor: Any,
        feature_frame: pd.DataFrame,
    ) -> list[float] | None:
        """Return positive-class probabilities when the model exposes them."""

        if not hasattr(predictor, "predict_proba"):
            return None

        probabilities = predictor.predict_proba(feature_frame)
        return [float(probability) for probability in probabilities[:, 1]]
