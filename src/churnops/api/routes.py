"""HTTP routes for the ChurnOps inference API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from churnops import __version__
from churnops.api.dependencies import get_inference_service
from churnops.api.schemas import (
    FeatureSchemaResponse,
    HealthResponse,
    ModelMetadataResponse,
    ModelReferenceResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResultResponse,
)
from churnops.inference import InferenceService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health(service: InferenceService = Depends(get_inference_service)) -> HealthResponse:
    """Return a lightweight health response for probes and load balancers."""

    health = service.get_health()
    return HealthResponse(
        service="churnops-inference-api",
        version=__version__,
        **health,
    )


@router.get("/v1/model/metadata", response_model=ModelMetadataResponse)
def get_model_metadata(
    service: InferenceService = Depends(get_inference_service),
) -> ModelMetadataResponse:
    """Return metadata describing the currently loaded model."""

    loaded_model = service.get_model_metadata()
    descriptor = loaded_model.descriptor
    return ModelMetadataResponse(
        model=ModelReferenceResponse(
            model_name=descriptor.model_name,
            source_type=descriptor.source_type,
            source_uri=descriptor.source_uri,
            loaded_at_utc=descriptor.loaded_at_utc,
            training_run_id=descriptor.training_run_id,
            registered_model_name=descriptor.registered_model_name,
            registered_model_version=descriptor.registered_model_version,
        ),
        positive_class_label=descriptor.positive_class_label,
        negative_class_label=descriptor.negative_class_label,
        prediction_threshold=descriptor.prediction_threshold,
        feature_schema=FeatureSchemaResponse(
            numeric_features=descriptor.numeric_features,
            categorical_features=descriptor.categorical_features,
        ),
    )


@router.post("/v1/predictions", response_model=PredictionResponse)
def predict_churn(
    request: PredictionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Run batch churn prediction against the configured model."""

    loaded_model, predictions = service.predict(
        [instance.model_dump(mode="python") for instance in request.instances]
    )
    descriptor = loaded_model.descriptor
    return PredictionResponse(
        model=ModelReferenceResponse(
            model_name=descriptor.model_name,
            source_type=descriptor.source_type,
            source_uri=descriptor.source_uri,
            loaded_at_utc=descriptor.loaded_at_utc,
            training_run_id=descriptor.training_run_id,
            registered_model_name=descriptor.registered_model_name,
            registered_model_version=descriptor.registered_model_version,
        ),
        predictions=[
            PredictionResultResponse(
                index=prediction.index,
                predicted_class=prediction.predicted_class,
                predicted_churn=prediction.predicted_churn,
                churn_probability=prediction.churn_probability,
                decision_threshold=descriptor.prediction_threshold,
            )
            for prediction in predictions
        ],
    )
