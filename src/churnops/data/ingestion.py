"""Raw dataset ingestion for the churn training workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churnops.config import DatasetConfig


def read_raw_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Read the configured churn dataset from local storage."""

    dataset_path = Path(config.raw_data_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataframe = pd.read_csv(
        dataset_path,
        na_values=config.na_values or None,
        keep_default_na=True,
    )
    return _standardize_columns(dataframe, config)


def load_raw_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Read and validate the configured churn dataset."""

    from churnops.data.validation import validate_raw_dataset

    dataframe = read_raw_dataset(config)
    validate_raw_dataset(dataframe, config)
    return dataframe


def _standardize_columns(dataframe: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Normalize raw column names and apply configured rename aliases."""

    standardized = dataframe.copy()
    standardized.columns = (
        standardized.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    )

    if config.column_renames:
        standardized = standardized.rename(columns=config.column_renames)

    return standardized
