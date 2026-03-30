"""Request and response schemas for the inference API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

NonEmptyString = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class ChurnPredictionInput(BaseModel):
    """Single churn prediction input row."""

    model_config = ConfigDict(extra="forbid")

    gender: NonEmptyString
    SeniorCitizen: Literal[0, 1]
    Partner: NonEmptyString
    Dependents: NonEmptyString
    tenure: Annotated[int, Field(ge=0)]
    PhoneService: NonEmptyString
    MultipleLines: NonEmptyString
    InternetService: NonEmptyString
    OnlineSecurity: NonEmptyString
    OnlineBackup: NonEmptyString
    DeviceProtection: NonEmptyString
    TechSupport: NonEmptyString
    StreamingTV: NonEmptyString
    StreamingMovies: NonEmptyString
    Contract: NonEmptyString
    PaperlessBilling: NonEmptyString
    PaymentMethod: NonEmptyString
    MonthlyCharges: Annotated[float, Field(ge=0)]
    TotalCharges: Annotated[float | None, Field(ge=0)] = None

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def normalize_total_charges(cls, value):
        """Allow numeric values or blank input for TotalCharges."""

        if value is None:
            return None

        if isinstance(value, str):
            trimmed_value = value.strip()
            if trimmed_value == "":
                return None
            try:
                return float(trimmed_value)
            except ValueError as error:
                raise ValueError("TotalCharges must be numeric or blank.") from error

        return float(value)


class PredictionRequest(BaseModel):
    """Batch prediction request."""

    model_config = ConfigDict(extra="forbid")

    instances: Annotated[list[ChurnPredictionInput], Field(min_length=1, max_length=1000)]


class ModelReferenceResponse(BaseModel):
    """Common model reference fields returned by API responses."""

    model_name: str
    source_type: str
    source_uri: str
    loaded_at_utc: datetime
    training_run_id: str | None = None
    registered_model_name: str | None = None
    registered_model_version: str | None = None


class PredictionResultResponse(BaseModel):
    """Prediction output for a single inference row."""

    index: int
    predicted_class: Literal[0, 1]
    predicted_churn: bool
    churn_probability: float | None
    decision_threshold: float


class PredictionResponse(BaseModel):
    """Batch prediction response."""

    model: ModelReferenceResponse
    predictions: list[PredictionResultResponse]


class HealthResponse(BaseModel):
    """Health response for the inference service."""

    status: Literal["ok", "degraded"]
    service: str
    version: str
    model_loaded: bool
    model_source: str
    last_error: str | None = None


class FeatureSchemaResponse(BaseModel):
    """Feature groups expected by the current model."""

    numeric_features: list[str]
    categorical_features: list[str]


class ModelMetadataResponse(BaseModel):
    """Metadata describing the currently loaded inference model."""

    model: ModelReferenceResponse
    positive_class_label: str
    negative_class_label: str | None
    prediction_threshold: float
    feature_schema: FeatureSchemaResponse
