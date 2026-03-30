"""Typed configuration models for the ChurnOps training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    column_renames: dict[str, str] = field(default_factory=dict)
    id_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    numeric_coercion_columns: list[str] = field(default_factory=list)
    na_values: list[str] = field(default_factory=list)
    infer_remaining_features: bool = False


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
    model_directory: str
    model_filename: str
    metrics_directory: str
    metrics_filename: str
    metadata_directory: str
    metadata_filename: str
    validation_report_filename: str
    config_directory: str
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
