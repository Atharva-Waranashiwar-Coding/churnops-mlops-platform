"""Raw dataset ingestion for the churn training workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churnops.config import DatasetConfig


def load_raw_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Load the configured churn dataset from local storage."""

    dataset_path = Path(config.raw_data_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataframe = pd.read_csv(
        dataset_path,
        na_values=config.na_values or None,
        keep_default_na=True,
    )
    dataframe.columns = dataframe.columns.str.strip()

    if dataframe.empty:
        raise ValueError(f"Dataset file is empty: {dataset_path}")

    _validate_required_columns(dataframe, config)
    return dataframe


def _validate_required_columns(dataframe: pd.DataFrame, config: DatasetConfig) -> None:
    """Ensure the raw dataset contains the configured contract columns."""

    if config.target_column not in dataframe.columns:
        raise ValueError(f"Target column '{config.target_column}' was not found in the dataset.")

    required_columns = set(config.required_columns)
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )
