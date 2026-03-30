"""Dataset validation utilities for the churn training workflow."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from churnops.config import DatasetConfig


@dataclass(slots=True)
class DatasetValidationReport:
    """Summary of a validated raw dataset contract."""

    row_count: int
    column_count: int
    validated_columns: list[str]
    target_distribution: dict[str, int]


def validate_raw_dataset(
    dataframe: pd.DataFrame,
    config: DatasetConfig,
) -> DatasetValidationReport:
    """Validate that a raw dataset satisfies the configured training contract."""

    if dataframe.empty:
        raise ValueError("Dataset is empty after ingestion.")

    duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(
            "Dataset contains duplicate column names after normalization: "
            + ", ".join(sorted(set(duplicate_columns)))
        )

    _ensure_target_column_present(dataframe, config.target_column)
    _ensure_columns_present(config.required_columns, dataframe, "data.required_columns")
    _ensure_columns_present(config.id_columns, dataframe, "data.id_columns")
    _ensure_columns_present(config.drop_columns, dataframe, "data.drop_columns")
    _ensure_columns_present(config.numeric_features, dataframe, "data.numeric_features")
    _ensure_columns_present(
        config.categorical_features,
        dataframe,
        "data.categorical_features",
    )
    _ensure_columns_present(
        config.numeric_coercion_columns,
        dataframe,
        "data.numeric_coercion_columns",
    )

    target_series = dataframe[config.target_column]
    if target_series.isna().any():
        raise ValueError("Target column contains missing values.")
    if target_series.nunique(dropna=True) < 2:
        raise ValueError("Target column must contain at least two classes.")

    distribution = (
        target_series.astype("string").value_counts(dropna=False).sort_index().to_dict()
    )

    return DatasetValidationReport(
        row_count=int(dataframe.shape[0]),
        column_count=int(dataframe.shape[1]),
        validated_columns=sorted(dataframe.columns.tolist()),
        target_distribution={str(label): int(count) for label, count in distribution.items()},
    )


def _ensure_columns_present(
    expected_columns: list[str],
    dataframe: pd.DataFrame,
    section_name: str,
) -> None:
    """Raise a clear error when configured columns are missing from a dataset."""

    missing_columns = sorted(set(expected_columns).difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"Configured columns in '{section_name}' were not found: "
            + ", ".join(missing_columns)
        )


def _ensure_target_column_present(dataframe: pd.DataFrame, target_column: str) -> None:
    """Raise a clear error when the configured target column is missing."""

    if target_column not in dataframe.columns:
        available_columns = ", ".join(dataframe.columns.tolist())
        raise ValueError(
            f"Target column '{target_column}' was not found in the dataset. "
            f"Available columns: {available_columns}"
        )
