"""Tests for raw dataset contract validation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from churnops.data.ingestion import read_raw_dataset
from churnops.data.validation import validate_raw_dataset


def test_validate_raw_dataset_returns_summary_report(dataset_config) -> None:
    """Validation should return a basic dataset summary for downstream metadata."""

    raw_dataset = read_raw_dataset(dataset_config)

    report = validate_raw_dataset(raw_dataset, dataset_config)

    assert report.row_count == 24
    assert report.column_count == 21
    assert report.target_distribution == {"No": 12, "Yes": 12}


def test_validate_raw_dataset_rejects_single_class_target(dataset_config) -> None:
    """Validation should reject datasets that cannot support binary training."""

    raw_dataset = read_raw_dataset(dataset_config)
    raw_dataset["Churn"] = "Yes"

    with pytest.raises(ValueError, match="at least two classes"):
        validate_raw_dataset(raw_dataset, dataset_config)


def test_validate_raw_dataset_rejects_missing_configured_feature(
    dataset_config,
) -> None:
    """Validation should fail early when configured feature columns are absent."""

    raw_dataset = read_raw_dataset(dataset_config).drop(columns=["tenure"])
    broken_config = replace(dataset_config, required_columns=["customerID", "Churn"])

    with pytest.raises(ValueError, match="data.numeric_features"):
        validate_raw_dataset(raw_dataset, broken_config)
