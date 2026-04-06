"""Internal models for the inference service layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelDescriptor:
    """Metadata describing the currently loaded inference model."""

    model_name: str
    source_type: str
    source_uri: str
    positive_class_label: str
    negative_class_label: str | None
    prediction_threshold: float
    loaded_at_utc: datetime
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    training_run_id: str | None = None
    registered_model_name: str | None = None
    registered_model_version: str | None = None
    local_run_directory: Path | None = None


@dataclass(slots=True)
class LoadedModel:
    """Predictor object plus its resolved metadata."""

    predictor: Any
    descriptor: ModelDescriptor


@dataclass(slots=True)
class PredictionRecord:
    """Prediction output for a single request instance."""

    index: int
    predicted_class: int
    predicted_churn: bool
    churn_probability: float | None
