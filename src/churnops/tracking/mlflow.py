"""MLflow-backed training run tracking."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import inspect
from typing import Any, Iterator

from churnops.config import ModelRegistryConfig, Settings
from churnops.tracking.models import CompletedTrainingRun, ModelRegistryResult, TrackingResult


class MLflowTrainingTracker:
    """Track completed training runs in MLflow."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        try:
            import mlflow
            import mlflow.sklearn
            from mlflow import MlflowClient
            from mlflow.exceptions import MlflowException
            from mlflow.models import infer_signature
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "MLflow tracking is enabled but the 'mlflow' package is not installed. "
                "Install the project dependencies before running training with tracking enabled."
            ) from error

        self._mlflow = mlflow
        self._mlflow_sklearn = mlflow.sklearn
        self._client_type = MlflowClient
        self._mlflow_exception = MlflowException
        self._infer_signature = infer_signature
        self._client: Any | None = None
        self._active_run_id: str | None = None
        self._active_run_name: str | None = None
        self._artifact_uri: str | None = None

    @contextmanager
    def start_run(self) -> Iterator[None]:
        """Start and manage an MLflow run for the pipeline execution."""

        self._mlflow.set_tracking_uri(self._settings.tracking.tracking_uri)
        if self._settings.tracking.registry_uri is not None:
            self._mlflow.set_registry_uri(self._settings.tracking.registry_uri)

        self._client = self._client_type(
            tracking_uri=self._settings.tracking.tracking_uri,
            registry_uri=self._settings.tracking.registry_uri,
        )
        experiment_id = self._ensure_experiment()
        run_name = self._build_run_name()
        run_tags = self._build_run_tags()

        with self._mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=run_tags,
        ) as run:
            self._active_run_id = run.info.run_id
            self._active_run_name = run_name
            self._artifact_uri = run.info.artifact_uri
            yield

    def finalize_run(self, completed_run: CompletedTrainingRun) -> TrackingResult:
        """Log params, metrics, artifacts, and optionally register the best model."""

        if self._active_run_id is None:
            raise RuntimeError("Cannot finalize tracking without an active MLflow run.")
        self._log_params(completed_run)
        self._log_metrics(completed_run)
        self._log_local_artifacts(completed_run)
        model_uri = self._log_model_artifact(completed_run)
        model_registry_result = self._register_best_model(completed_run, model_uri)

        return TrackingResult(
            enabled=True,
            backend="mlflow",
            experiment_name=self._settings.tracking.experiment_name,
            run_id=self._active_run_id,
            run_name=self._active_run_name,
            tracking_uri=self._settings.tracking.tracking_uri,
            registry_uri=self._settings.tracking.registry_uri,
            artifact_uri=self._artifact_uri,
            model_registry=model_registry_result,
        )

    def _build_run_name(self) -> str:
        """Create a stable, meaningful MLflow run name."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = self._settings.tracking.run_name_prefix or self._settings.project.name
        return f"{prefix}-{self._settings.model.name}-{timestamp}"

    def _build_run_tags(self) -> dict[str, str]:
        """Build the baseline tag set for tracked runs."""

        tags = {
            "churnops.entrypoint": "local_training",
            "churnops.model_name": self._settings.model.name,
            "churnops.project_name": self._settings.project.name,
            "churnops.target_column": self._settings.data.target_column,
        }
        if self._settings.tracking.model_registry.enabled:
            tags["churnops.registry.metric"] = (
                f"{self._settings.tracking.model_registry.comparison_split}."
                f"{self._settings.tracking.model_registry.comparison_metric}"
            )
        tags.update(self._settings.tracking.tags)
        return tags

    def _ensure_experiment(self) -> str:
        """Create the configured experiment with an explicit artifact location if needed."""

        if self._client is None:
            raise RuntimeError("MLflow client is not initialized.")

        experiment = self._client.get_experiment_by_name(self._settings.tracking.experiment_name)
        if experiment is not None:
            return str(experiment.experiment_id)

        return str(
            self._client.create_experiment(
                name=self._settings.tracking.experiment_name,
                artifact_location=self._settings.tracking.artifact_location,
            )
        )

    def _log_params(self, completed_run: CompletedTrainingRun) -> None:
        """Log flattened run parameters for searchability in MLflow."""

        params = {
            "project.name": completed_run.settings.project.name,
            "data.raw_data_path": str(completed_run.settings.data.raw_data_path),
            "data.target_column": completed_run.settings.data.target_column,
            "data.positive_class": completed_run.settings.data.positive_class,
            "data.row_count": str(completed_run.validation_report.row_count),
            "data.column_count": str(completed_run.validation_report.column_count),
            "data.numeric_feature_count": str(
                len(completed_run.trained_model.feature_spec.numeric_features)
            ),
            "data.categorical_feature_count": str(
                len(completed_run.trained_model.feature_spec.categorical_features)
            ),
            "split.train_rows": str(completed_run.evaluation_result.split_sizes["train"]),
            "split.test_rows": str(completed_run.evaluation_result.split_sizes["test"]),
            "split.test_size": str(completed_run.settings.split.test_size),
            "split.validation_size": str(completed_run.settings.split.validation_size),
            "split.random_state": str(completed_run.settings.split.random_state),
            "model.name": completed_run.settings.model.name,
        }
        if "validation" in completed_run.evaluation_result.split_sizes:
            params["split.validation_rows"] = str(
                completed_run.evaluation_result.split_sizes["validation"]
            )
        params.update(
            {
                f"model.param.{param_name}": str(param_value)
                for param_name, param_value in completed_run.settings.model.params.items()
            }
        )
        self._mlflow.log_params(params)

    def _log_metrics(self, completed_run: CompletedTrainingRun) -> None:
        """Log per-split evaluation metrics into MLflow."""

        for split_name, split_metrics in completed_run.evaluation_result.metrics.items():
            for metric_name, metric_value in split_metrics.items():
                if metric_value is None:
                    continue
                self._mlflow.log_metric(f"{split_name}_{metric_name}", float(metric_value))

    def _log_local_artifacts(self, completed_run: CompletedTrainingRun) -> None:
        """Log the persisted local run bundle to MLflow."""

        self._mlflow.log_artifacts(
            str(completed_run.persisted_run.run_directory),
            artifact_path=self._settings.tracking.local_artifacts_path,
        )

    def _log_model_artifact(self, completed_run: CompletedTrainingRun) -> str:
        """Log the trained sklearn pipeline using the MLflow sklearn flavor."""

        input_example = completed_run.data_splits.X_train.head(5)
        predictions = completed_run.trained_model.model_pipeline.predict(input_example)
        signature = self._infer_signature(input_example, predictions)
        log_model_parameters = inspect.signature(self._mlflow_sklearn.log_model).parameters
        log_model_kwargs = {
            "sk_model": completed_run.trained_model.model_pipeline,
            "input_example": input_example,
            "signature": signature,
        }
        if "name" in log_model_parameters:
            log_model_kwargs["name"] = self._settings.tracking.model_artifact_path
        else:
            log_model_kwargs["artifact_path"] = self._settings.tracking.model_artifact_path

        model_info = self._mlflow_sklearn.log_model(**log_model_kwargs)
        return getattr(
            model_info,
            "model_uri",
            f"runs:/{self._active_run_id}/{self._settings.tracking.model_artifact_path}",
        )

    def _register_best_model(
        self,
        completed_run: CompletedTrainingRun,
        model_uri: str,
    ) -> ModelRegistryResult:
        """Register the current model only when it beats the existing incumbent."""

        registry_config = self._settings.tracking.model_registry
        if not registry_config.enabled:
            self._mlflow.set_tag("churnops.registry.status", "disabled")
            return ModelRegistryResult(
                attempted=False,
                registered=False,
                status="disabled",
            )

        candidate_metric = self._resolve_candidate_metric(
            completed_run,
            registry_config,
        )
        if candidate_metric is None:
            self._mlflow.set_tag("churnops.registry.status", "metric_unavailable")
            return ModelRegistryResult(
                attempted=True,
                registered=False,
                status="metric_unavailable",
                model_name=registry_config.model_name,
                metric_name=registry_config.comparison_metric,
                metric_split=registry_config.comparison_split,
            )

        incumbent = self._find_best_registered_metric(registry_config)
        if incumbent is not None and not self._is_better(
            candidate_metric,
            incumbent["metric"],
            registry_config.greater_is_better,
        ):
            self._mlflow.set_tags(
                {
                    "churnops.registry.status": "not_best",
                    "churnops.registry.model_name": str(registry_config.model_name),
                }
            )
            return ModelRegistryResult(
                attempted=True,
                registered=False,
                status="not_best",
                model_name=registry_config.model_name,
                metric_name=registry_config.comparison_metric,
                metric_split=registry_config.comparison_split,
                candidate_metric=candidate_metric,
                incumbent_metric=incumbent["metric"],
                incumbent_version=incumbent["version"],
            )

        self._ensure_registered_model(registry_config.model_name)
        model_version = self._mlflow.register_model(
            model_uri=model_uri,
            name=registry_config.model_name,
        )
        self._tag_registered_model_version(
            registry_config=registry_config,
            model_version=str(model_version.version),
            candidate_metric=candidate_metric,
        )
        self._mlflow.set_tags(
            {
                "churnops.registry.status": "registered",
                "churnops.registry.model_name": str(registry_config.model_name),
                "churnops.registry.model_version": str(model_version.version),
            }
        )
        return ModelRegistryResult(
            attempted=True,
            registered=True,
            status="registered",
            model_name=registry_config.model_name,
            model_version=str(model_version.version),
            metric_name=registry_config.comparison_metric,
            metric_split=registry_config.comparison_split,
            candidate_metric=candidate_metric,
            incumbent_metric=incumbent["metric"] if incumbent is not None else None,
            incumbent_version=incumbent["version"] if incumbent is not None else None,
        )

    def _resolve_candidate_metric(
        self,
        completed_run: CompletedTrainingRun,
        registry_config: ModelRegistryConfig,
    ) -> float | None:
        """Extract the configured comparison metric from the evaluation result."""

        split_metrics = completed_run.evaluation_result.metrics.get(registry_config.comparison_split)
        if split_metrics is None:
            return None

        metric_value = split_metrics.get(registry_config.comparison_metric)
        if metric_value is None:
            return None

        return float(metric_value)

    def _find_best_registered_metric(
        self,
        registry_config: ModelRegistryConfig,
    ) -> dict[str, float | str] | None:
        """Find the best registered model version using the configured comparison metric."""

        if self._client is None:
            raise RuntimeError("MLflow client is not initialized.")

        filter_expression = f"name = '{registry_config.model_name}'"
        try:
            versions = self._client.search_model_versions(filter_expression)
        except self._mlflow_exception:
            return None

        metric_key = (
            f"{registry_config.comparison_split}_{registry_config.comparison_metric}"
        )
        best: dict[str, float | str] | None = None

        for version in versions:
            if version.run_id == self._active_run_id:
                continue

            metric_value = self._extract_registered_metric(version, metric_key, registry_config)
            if metric_value is None:
                continue

            if best is None or self._is_better(
                metric_value,
                float(best["metric"]),
                registry_config.greater_is_better,
            ):
                best = {
                    "metric": metric_value,
                    "version": str(version.version),
                }

        return best

    def _extract_registered_metric(
        self,
        version: Any,
        metric_key: str,
        registry_config: ModelRegistryConfig,
    ) -> float | None:
        """Resolve the incumbent metric from version tags or the source run."""

        if self._client is None:
            raise RuntimeError("MLflow client is not initialized.")

        version_tags = version.tags or {}
        metric_tag = version_tags.get("churnops.registry.candidate_metric")
        split_tag = version_tags.get("churnops.registry.metric_split")
        name_tag = version_tags.get("churnops.registry.metric_name")
        if (
            metric_tag is not None
            and split_tag == registry_config.comparison_split
            and name_tag == registry_config.comparison_metric
        ):
            return float(metric_tag)

        if not version.run_id:
            return None

        run = self._client.get_run(version.run_id)
        run_metric = run.data.metrics.get(metric_key)
        if run_metric is None:
            return None

        return float(run_metric)

    def _ensure_registered_model(self, model_name: str | None) -> None:
        """Create the registered model shell if it does not exist yet."""

        if self._client is None:
            raise RuntimeError("MLflow client is not initialized.")
        if model_name is None:
            raise ValueError("Model registry is enabled but no model name was provided.")

        try:
            self._client.create_registered_model(model_name)
        except self._mlflow_exception as error:
            error_message = str(error)
            if "RESOURCE_ALREADY_EXISTS" not in error_message:
                raise

    def _tag_registered_model_version(
        self,
        registry_config: ModelRegistryConfig,
        model_version: str,
        candidate_metric: float,
    ) -> None:
        """Attach comparison metadata to the registered model version."""

        if self._client is None:
            raise RuntimeError("MLflow client is not initialized.")
        if registry_config.model_name is None:
            raise ValueError("Model registry is enabled but no model name was provided.")

        tags = {
            "churnops.registry.metric_name": registry_config.comparison_metric,
            "churnops.registry.metric_split": registry_config.comparison_split,
            "churnops.registry.candidate_metric": str(candidate_metric),
        }
        for tag_name, tag_value in tags.items():
            self._client.set_model_version_tag(
                name=registry_config.model_name,
                version=model_version,
                key=tag_name,
                value=tag_value,
            )

        if registry_config.alias:
            self._client.set_registered_model_alias(
                name=registry_config.model_name,
                alias=registry_config.alias,
                version=model_version,
            )

    @staticmethod
    def _is_better(candidate: float, incumbent: float, greater_is_better: bool) -> bool:
        """Return whether the candidate metric beats the incumbent."""

        if greater_is_better:
            return candidate > incumbent
        return candidate < incumbent
