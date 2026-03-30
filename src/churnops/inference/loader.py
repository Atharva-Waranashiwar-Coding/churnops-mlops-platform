"""Model loading helpers for inference execution."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib

from churnops.config import Settings
from churnops.inference.exceptions import ModelConfigurationError, ModelLoadError
from churnops.inference.models import LoadedModel, ModelDescriptor


def load_inference_model(settings: Settings) -> LoadedModel:
    """Load the configured inference model from the selected backend."""

    source = settings.inference.model_source
    if source == "local_artifact":
        return _load_from_local_artifact(settings)
    if source == "mlflow_model_uri":
        return _load_from_mlflow_model_uri(settings)
    if source == "mlflow_registry":
        return _load_from_mlflow_registry(settings)

    raise ModelConfigurationError(f"Unsupported inference.model_source value: {source}")


def _load_from_local_artifact(settings: Settings) -> LoadedModel:
    """Load a persisted sklearn pipeline from the local artifact store."""

    model_path, run_directory = _resolve_local_model_artifact(settings)
    if not model_path.exists():
        raise ModelLoadError(f"Configured local model artifact was not found: {model_path}")

    predictor = joblib.load(model_path)
    run_metadata = _read_json_if_present(
        run_directory / settings.artifacts.metadata_directory / settings.artifacts.metadata_filename
        if run_directory is not None
        else None
    )
    validation_report = _read_json_if_present(
        run_directory
        / settings.artifacts.metadata_directory
        / settings.artifacts.validation_report_filename
        if run_directory is not None
        else None
    )
    return LoadedModel(
        predictor=predictor,
        descriptor=_build_local_descriptor(
            settings=settings,
            model_path=model_path,
            run_metadata=run_metadata,
            validation_report=validation_report,
        ),
    )


def _load_from_mlflow_model_uri(settings: Settings) -> LoadedModel:
    """Load a model from an explicit MLflow model URI."""

    if settings.inference.model_uri is None:
        raise ModelConfigurationError(
            "inference.model_uri must be set for 'mlflow_model_uri' inference."
        )

    mlflow_sklearn, client = _load_mlflow_clients(settings)
    predictor = mlflow_sklearn.load_model(settings.inference.model_uri)
    run_id = _extract_run_id_from_model_uri(settings.inference.model_uri)
    run = client.get_run(run_id) if run_id is not None else None
    return LoadedModel(
        predictor=predictor,
        descriptor=_build_mlflow_descriptor(
            settings=settings,
            source_type="mlflow_model_uri",
            source_uri=settings.inference.model_uri,
            run=run,
            training_run_id=run_id,
        ),
    )


def _load_from_mlflow_registry(settings: Settings) -> LoadedModel:
    """Load a model from the MLflow model registry using alias or version pinning."""

    mlflow_sklearn, client = _load_mlflow_clients(settings)
    model_uri, version_info = _resolve_registry_model_reference(settings, client)
    predictor = mlflow_sklearn.load_model(model_uri)
    run = client.get_run(version_info.run_id) if version_info.run_id else None
    return LoadedModel(
        predictor=predictor,
        descriptor=_build_mlflow_descriptor(
            settings=settings,
            source_type="mlflow_registry",
            source_uri=model_uri,
            run=run,
            training_run_id=version_info.run_id,
            registered_model_name=version_info.name,
            registered_model_version=str(version_info.version),
        ),
    )


def _resolve_local_model_artifact(settings: Settings) -> tuple[Path, Path | None]:
    """Resolve the local model path and its run directory when available."""

    inference_settings = settings.inference
    if inference_settings.local_model_path is not None:
        model_path = inference_settings.local_model_path
        run_directory = _infer_run_directory_from_model_path(settings, model_path)
        return model_path, run_directory

    training_root = settings.artifacts.root_dir / settings.artifacts.training_runs_dir
    if inference_settings.local_run_id is not None:
        run_directory = training_root / inference_settings.local_run_id
        model_path = run_directory / settings.artifacts.model_directory / settings.artifacts.model_filename
        return model_path, run_directory

    if not training_root.exists():
        raise ModelLoadError(
            "No local training artifacts were found. Train a model first or configure a different "
            "inference model source."
        )

    candidate_runs = sorted(
        (
            run_directory
            for run_directory in training_root.iterdir()
            if run_directory.is_dir()
            and (
                run_directory
                / settings.artifacts.model_directory
                / settings.artifacts.model_filename
            ).exists()
        ),
        key=lambda run_directory: run_directory.name,
        reverse=True,
    )
    if not candidate_runs:
        raise ModelLoadError(
            "No persisted local model artifacts were found. Train a model first or configure "
            "inference.local_model_path."
        )

    run_directory = candidate_runs[0]
    model_path = run_directory / settings.artifacts.model_directory / settings.artifacts.model_filename
    return model_path, run_directory


def _infer_run_directory_from_model_path(settings: Settings, model_path: Path) -> Path | None:
    """Infer the run directory from a configured local model path when it follows the standard layout."""

    if model_path.parent.name != settings.artifacts.model_directory:
        return None

    candidate_run_directory = model_path.parent.parent
    if (
        candidate_run_directory
        / settings.artifacts.metadata_directory
        / settings.artifacts.metadata_filename
    ).exists():
        return candidate_run_directory
    return None


def _build_local_descriptor(
    settings: Settings,
    model_path: Path,
    run_metadata: dict[str, Any] | None,
    validation_report: dict[str, Any] | None,
) -> ModelDescriptor:
    """Build the model descriptor for a locally persisted artifact."""

    numeric_features = _resolve_feature_list(run_metadata, settings, "numeric_features")
    categorical_features = _resolve_feature_list(run_metadata, settings, "categorical_features")
    return ModelDescriptor(
        model_name=str(run_metadata.get("model", {}).get("name", settings.model.name))
        if run_metadata is not None
        else settings.model.name,
        source_type="local_artifact",
        source_uri=str(model_path),
        positive_class_label=settings.data.positive_class,
        negative_class_label=_infer_negative_class_label(settings, validation_report),
        prediction_threshold=settings.inference.prediction_threshold,
        loaded_at_utc=datetime.now(timezone.utc),
        feature_names=[*numeric_features, *categorical_features],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        training_run_id=str(run_metadata.get("run_id")) if run_metadata is not None else None,
    )


def _build_mlflow_descriptor(
    settings: Settings,
    source_type: str,
    source_uri: str,
    run: Any | None,
    training_run_id: str | None,
    registered_model_name: str | None = None,
    registered_model_version: str | None = None,
) -> ModelDescriptor:
    """Build the model descriptor for an MLflow-loaded artifact."""

    numeric_features = list(settings.data.numeric_features)
    categorical_features = list(settings.data.categorical_features)
    model_name = settings.model.name
    if run is not None:
        model_name = str(run.data.params.get("model.name", model_name))

    return ModelDescriptor(
        model_name=model_name,
        source_type=source_type,
        source_uri=source_uri,
        positive_class_label=settings.data.positive_class,
        negative_class_label=None,
        prediction_threshold=settings.inference.prediction_threshold,
        loaded_at_utc=datetime.now(timezone.utc),
        feature_names=[*numeric_features, *categorical_features],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        training_run_id=training_run_id,
        registered_model_name=registered_model_name,
        registered_model_version=registered_model_version,
    )


def _resolve_registry_model_reference(settings: Settings, client: Any) -> tuple[str, Any]:
    """Build a registry model URI from the configured alias or version pin."""

    inference_settings = settings.inference
    if inference_settings.registered_model_name is None:
        raise ModelConfigurationError(
            "inference.registered_model_name must be set for 'mlflow_registry' inference."
        )

    if inference_settings.registered_model_alias is not None:
        version_info = client.get_model_version_by_alias(
            name=inference_settings.registered_model_name,
            alias=inference_settings.registered_model_alias,
        )
        model_uri = (
            f"models:/{inference_settings.registered_model_name}"
            f"@{inference_settings.registered_model_alias}"
        )
        return model_uri, version_info

    if inference_settings.registered_model_version is not None:
        version_info = client.get_model_version(
            name=inference_settings.registered_model_name,
            version=inference_settings.registered_model_version,
        )
        model_uri = (
            f"models:/{inference_settings.registered_model_name}/"
            f"{inference_settings.registered_model_version}"
        )
        return model_uri, version_info

    raise ModelConfigurationError(
        "Configure inference.registered_model_alias or inference.registered_model_version for "
        "'mlflow_registry' inference."
    )


def _load_mlflow_clients(settings: Settings) -> tuple[Any, Any]:
    """Load MLflow client dependencies using the configured tracking backend."""

    try:
        import mlflow.sklearn
        from mlflow import MlflowClient
    except ModuleNotFoundError as error:
        raise ModelLoadError(
            "MLflow is required for the configured inference model source but is not installed."
        ) from error

    client = MlflowClient(
        tracking_uri=settings.tracking.tracking_uri,
        registry_uri=settings.tracking.registry_uri,
    )
    return mlflow.sklearn, client


def _extract_run_id_from_model_uri(model_uri: str) -> str | None:
    """Extract an MLflow run ID from a runs URI when possible."""

    if not model_uri.startswith("runs:/"):
        return None

    remainder = model_uri.removeprefix("runs:/")
    run_id, _, _ = remainder.partition("/")
    return run_id or None


def _resolve_feature_list(
    run_metadata: dict[str, Any] | None,
    settings: Settings,
    key: str,
) -> list[str]:
    """Resolve a feature list from run metadata when available, else fall back to settings."""

    metadata_features = run_metadata.get("data", {}).get(key) if run_metadata is not None else None
    if isinstance(metadata_features, list) and metadata_features:
        return [str(feature_name) for feature_name in metadata_features]

    return [str(feature_name) for feature_name in getattr(settings.data, key)]


def _infer_negative_class_label(
    settings: Settings,
    validation_report: dict[str, Any] | None,
) -> str | None:
    """Infer the negative class label from persisted validation metadata when available."""

    if validation_report is None:
        return None

    target_distribution = validation_report.get("target_distribution", {})
    if not isinstance(target_distribution, dict):
        return None

    for class_label in target_distribution.keys():
        if str(class_label) != settings.data.positive_class:
            return str(class_label)
    return None


def _read_json_if_present(path: Path | None) -> dict[str, Any] | None:
    """Read a JSON file when it exists, else return no metadata."""

    if path is None or not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict):
        return value
    return None
