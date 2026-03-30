"""Typed configuration loading for the ChurnOps training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProjectConfig:
    """Project-level metadata and path resolution settings."""

    name: str
    root_dir: Path


@dataclass(slots=True)
class DatasetConfig:
    """Dataset access and schema expectations for churn training."""

    raw_data_path: Path
    target_column: str
    positive_class: str
    id_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    numeric_coercion_columns: list[str] = field(default_factory=list)
    na_values: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SplitConfig:
    """Train, validation, and test split parameters."""

    test_size: float
    validation_size: float
    random_state: int

    def __post_init__(self) -> None:
        """Validate split ratios after deserialization."""
        if not 0 < self.test_size < 1:
            raise ValueError("split.test_size must be between 0 and 1.")
        if not 0 <= self.validation_size < 1:
            raise ValueError("split.validation_size must be between 0 and 1.")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("split.test_size and split.validation_size must sum to less than 1.")


@dataclass(slots=True)
class ModelConfig:
    """Model selection and estimator parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArtifactConfig:
    """Artifact persistence settings for a training run."""

    root_dir: Path
    training_runs_dir: str
    model_filename: str
    metrics_filename: str
    metadata_filename: str
    config_snapshot_filename: str


@dataclass(slots=True)
class Settings:
    """Top-level application settings for a training run."""

    project: ProjectConfig
    data: DatasetConfig
    split: SplitConfig
    model: ModelConfig
    artifacts: ArtifactConfig
    config_path: Path


def load_settings(config_path: str | Path) -> Settings:
    """Load and validate application settings from a YAML file."""

    resolved_config_path = Path(config_path).expanduser().resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as config_file:
        raw_settings = yaml.safe_load(config_file) or {}

    if not isinstance(raw_settings, dict):
        raise ValueError("Configuration file must contain a YAML mapping at the top level.")

    project_section = _as_mapping(raw_settings.get("project", {}), "project")
    project_root = Path(project_section.get("root_dir", "."))
    if not project_root.is_absolute():
        project_root = (resolved_config_path.parent / project_root).resolve()

    project = ProjectConfig(
        name=str(project_section.get("name", "churnops")),
        root_dir=project_root,
    )

    data_section = _as_mapping(raw_settings.get("data", {}), "data")
    split_section = _as_mapping(raw_settings.get("split", {}), "split")
    model_section = _as_mapping(raw_settings.get("model", {}), "model")
    artifacts_section = _as_mapping(raw_settings.get("artifacts", {}), "artifacts")

    dataset = DatasetConfig(
        raw_data_path=_resolve_path(project.root_dir, data_section["raw_data_path"]),
        target_column=str(data_section["target_column"]),
        positive_class=str(data_section.get("positive_class", "Yes")),
        id_columns=_as_string_list(data_section.get("id_columns", []), "data.id_columns"),
        drop_columns=_as_string_list(data_section.get("drop_columns", []), "data.drop_columns"),
        required_columns=_as_string_list(data_section.get("required_columns", []), "data.required_columns"),
        numeric_features=_as_string_list(data_section.get("numeric_features", []), "data.numeric_features"),
        categorical_features=_as_string_list(
            data_section.get("categorical_features", []),
            "data.categorical_features",
        ),
        numeric_coercion_columns=_as_string_list(
            data_section.get("numeric_coercion_columns", []),
            "data.numeric_coercion_columns",
        ),
        na_values=_as_string_list(data_section.get("na_values", []), "data.na_values"),
    )

    split = SplitConfig(
        test_size=float(split_section.get("test_size", 0.2)),
        validation_size=float(split_section.get("validation_size", 0.1)),
        random_state=int(split_section.get("random_state", 42)),
    )

    model = ModelConfig(
        name=str(model_section.get("name", "logistic_regression")),
        params=_as_mapping(model_section.get("params", {}), "model.params"),
    )

    artifacts = ArtifactConfig(
        root_dir=_resolve_path(project.root_dir, artifacts_section.get("root_dir", "artifacts")),
        training_runs_dir=str(artifacts_section.get("training_runs_dir", "training")),
        model_filename=str(artifacts_section.get("model_filename", "model.joblib")),
        metrics_filename=str(artifacts_section.get("metrics_filename", "metrics.json")),
        metadata_filename=str(artifacts_section.get("metadata_filename", "metadata.json")),
        config_snapshot_filename=str(
            artifacts_section.get("config_snapshot_filename", "config.yaml")
        ),
    )

    return Settings(
        project=project,
        data=dataset,
        split=split,
        model=model,
        artifacts=artifacts,
        config_path=resolved_config_path,
    )


def _resolve_path(base_dir: Path, candidate: str | Path) -> Path:
    """Resolve a potentially relative path against the configured project root."""

    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _as_mapping(value: Any, section_name: str) -> dict[str, Any]:
    """Ensure a configuration section is a mapping."""

    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{section_name}' must be a mapping.")
    return value


def _as_string_list(value: Any, section_name: str) -> list[str]:
    """Ensure a configuration field is a list of strings."""

    if not isinstance(value, list):
        raise ValueError(f"Configuration section '{section_name}' must be a list.")

    string_list = [str(item) for item in value]
    if len(set(string_list)) != len(string_list):
        raise ValueError(f"Configuration section '{section_name}' contains duplicate values.")
    return string_list
