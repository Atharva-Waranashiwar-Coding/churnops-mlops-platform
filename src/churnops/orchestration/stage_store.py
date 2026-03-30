"""Filesystem-backed storage for orchestrated training stage outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churnops.models.evaluation import EvaluationResult
from churnops.orchestration.models import TrainingExecutionContext


class TrainingStageStore:
    """Persist and reload stage outputs for task-oriented training execution."""

    def __init__(self, context: TrainingExecutionContext) -> None:
        self._context = context

    @property
    def context_path(self) -> Path:
        """Return the persisted execution-context path."""

        return self._context.workspace_dir / "context.json"

    @property
    def raw_dataset_path(self) -> Path:
        """Return the normalized raw-dataset path."""

        return self.stage_dir("ingestion") / "raw_dataset.joblib"

    @property
    def validation_report_path(self) -> Path:
        """Return the dataset validation-report path."""

        return self.stage_dir("validation") / "validation_report.json"

    @property
    def prepared_dataset_path(self) -> Path:
        """Return the prepared dataset artifact path."""

        return self.stage_dir("preprocessing") / "prepared_dataset.joblib"

    @property
    def data_splits_path(self) -> Path:
        """Return the train/validation/test split artifact path."""

        return self.stage_dir("preprocessing") / "data_splits.joblib"

    @property
    def trained_model_path(self) -> Path:
        """Return the trained model artifact path."""

        return self.stage_dir("training") / "trained_model.joblib"

    @property
    def evaluation_result_path(self) -> Path:
        """Return the evaluation result artifact path."""

        return self.stage_dir("evaluation") / "evaluation_result.joblib"

    @property
    def evaluation_metrics_path(self) -> Path:
        """Return the JSON metrics summary path."""

        return self.stage_dir("evaluation") / "metrics.json"

    @property
    def registration_summary_path(self) -> Path:
        """Return the final model-publication summary path."""

        return self.stage_dir("registration") / "summary.json"

    def initialize(self) -> None:
        """Create the workspace skeleton for an orchestrated execution."""

        self._context.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.write_json(self.context_path, self._context.to_payload())

    def stage_dir(self, stage_name: str) -> Path:
        """Return and ensure a stage directory."""

        stage_directory = self._context.workspace_dir / stage_name
        stage_directory.mkdir(parents=True, exist_ok=True)
        return stage_directory

    def save_joblib(self, target_path: Path, payload: Any) -> Path:
        """Persist a Python object via joblib."""

        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, target_path)
        return target_path

    def load_joblib(self, target_path: Path) -> Any:
        """Load a joblib-backed stage artifact."""

        return joblib.load(target_path)

    def save_dataframe(self, dataframe: pd.DataFrame) -> Path:
        """Persist the normalized raw dataset."""

        return self.save_joblib(self.raw_dataset_path, dataframe)

    def load_dataframe(self) -> pd.DataFrame:
        """Load the normalized raw dataset."""

        return self.load_joblib(self.raw_dataset_path)

    def save_validation_report(self, report: Any) -> Path:
        """Persist a validation report as JSON."""

        return self.write_json(self.validation_report_path, self._serialize(report))

    def load_validation_report(self) -> Any:
        """Load a validation report from JSON."""

        from churnops.data.validation import DatasetValidationReport

        return DatasetValidationReport(**self.read_json(self.validation_report_path))

    def save_prepared_dataset(self, prepared_dataset: Any) -> Path:
        """Persist the prepared dataset object."""

        return self.save_joblib(self.prepared_dataset_path, prepared_dataset)

    def load_prepared_dataset(self) -> Any:
        """Load the prepared dataset object."""

        return self.load_joblib(self.prepared_dataset_path)

    def save_data_splits(self, data_splits: Any) -> Path:
        """Persist the split datasets object."""

        return self.save_joblib(self.data_splits_path, data_splits)

    def load_data_splits(self) -> Any:
        """Load the split datasets object."""

        return self.load_joblib(self.data_splits_path)

    def save_trained_model(self, trained_model: Any) -> Path:
        """Persist the trained model wrapper."""

        return self.save_joblib(self.trained_model_path, trained_model)

    def load_trained_model(self) -> Any:
        """Load the trained model wrapper."""

        return self.load_joblib(self.trained_model_path)

    def save_evaluation_result(self, evaluation_result: EvaluationResult) -> Path:
        """Persist the evaluation result as joblib and JSON."""

        self.save_joblib(self.evaluation_result_path, evaluation_result)
        self.write_json(self.evaluation_metrics_path, self._serialize(evaluation_result))
        return self.evaluation_result_path

    def load_evaluation_result(self) -> EvaluationResult:
        """Load the evaluation result."""

        return self.load_joblib(self.evaluation_result_path)

    def write_registration_summary(self, payload: dict[str, Any]) -> Path:
        """Persist the final publication summary."""

        return self.write_json(self.registration_summary_path, payload)

    def write_json(self, target_path: Path, payload: Any) -> Path:
        """Persist a JSON-serializable payload."""

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as target_file:
            json.dump(payload, target_file, indent=2, sort_keys=True)
        return target_path

    def read_json(self, target_path: Path) -> dict[str, Any]:
        """Load a JSON payload from disk."""

        with target_path.open("r", encoding="utf-8") as target_file:
            return json.load(target_file)

    def _serialize(self, payload: Any) -> Any:
        """Serialize supported dataclasses into JSON-friendly payloads."""

        if is_dataclass(payload):
            return asdict(payload)
        return payload
