"""Shared test fixtures for the ChurnOps baseline training workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from churnops.config import DatasetConfig

FIXTURE_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


@pytest.fixture()
def churn_fixture_path() -> Path:
    """Return the path to the churn CSV fixture used in tests."""

    return Path(__file__).parent / "fixtures" / "customer_churn.csv"


@pytest.fixture()
def dataset_config(churn_fixture_path: Path) -> DatasetConfig:
    """Return a dataset configuration compatible with the churn CSV fixture."""

    return DatasetConfig(
        raw_data_path=churn_fixture_path,
        target_column="Churn",
        positive_class="Yes",
        column_renames={},
        id_columns=["customerID"],
        drop_columns=[],
        required_columns=FIXTURE_COLUMNS,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        numeric_coercion_columns=["TotalCharges"],
        na_values=["", " "],
        infer_remaining_features=False,
    )
