"""Tests for raw churn dataset ingestion."""

from __future__ import annotations

from dataclasses import replace

import pytest

from churnops.data.ingestion import load_raw_dataset


def test_load_raw_dataset_reads_csv_fixture(dataset_config) -> None:
    """The ingestion layer should read a valid churn CSV fixture."""

    dataset = load_raw_dataset(dataset_config)

    assert dataset.shape == (24, 21)
    assert dataset.columns[0] == "customerID"
    assert dataset["Churn"].isin(["Yes", "No"]).all()


def test_load_raw_dataset_raises_when_target_column_is_missing(
    dataset_config,
    tmp_path,
) -> None:
    """The ingestion layer should reject CSV files without the configured target."""

    broken_dataset_path = tmp_path / "broken_dataset.csv"
    broken_dataset_path.write_text(
        "customerID,MonthlyCharges\n0001-AAAAA,29.85\n",
        encoding="utf-8",
    )

    broken_config = replace(dataset_config, raw_data_path=broken_dataset_path)

    with pytest.raises(ValueError, match="Target column 'Churn'"):
        load_raw_dataset(broken_config)


def test_load_raw_dataset_applies_configured_column_renames(
    dataset_config,
    tmp_path,
) -> None:
    """The ingestion layer should rename raw schema variants into canonical columns."""

    renamed_dataset_path = tmp_path / "renamed_dataset.csv"
    renamed_dataset_path.write_text(
        "CustomerID,Gender,Churn Label\n0001-AAAAA,Female,Yes\n",
        encoding="utf-8",
    )

    renamed_config = replace(
        dataset_config,
        raw_data_path=renamed_dataset_path,
        required_columns=["customerID", "gender", "Churn"],
        id_columns=["customerID"],
        numeric_features=[],
        categorical_features=["gender"],
        numeric_coercion_columns=[],
        column_renames={
            "CustomerID": "customerID",
            "Gender": "gender",
            "Churn Label": "Churn",
        },
    )

    dataset = load_raw_dataset(renamed_config)

    assert list(dataset.columns) == ["customerID", "gender", "Churn"]
