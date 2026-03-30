"""Tests for churn preprocessing and dataset splitting."""

from __future__ import annotations

import pandas as pd

from churnops.config import SplitConfig
from churnops.data.ingestion import load_raw_dataset
from churnops.features.preprocessing import (
    build_preprocessor,
    prepare_training_dataset,
    split_dataset,
)


def test_prepare_training_dataset_encodes_target_and_drops_identifier(
    dataset_config,
) -> None:
    """Prepared training data should exclude identifiers and encode the target."""

    raw_dataset = load_raw_dataset(dataset_config)
    prepared_dataset = prepare_training_dataset(raw_dataset, dataset_config)

    assert "customerID" not in prepared_dataset.features.columns
    assert set(prepared_dataset.target.unique()) == {0, 1}
    assert pd.api.types.is_numeric_dtype(prepared_dataset.features["TotalCharges"])
    assert prepared_dataset.feature_spec.numeric_features == [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]


def test_build_preprocessor_and_split_dataset_create_trainable_outputs(
    dataset_config,
) -> None:
    """The preprocessing layer should emit non-empty transformed matrices and splits."""

    raw_dataset = load_raw_dataset(dataset_config)
    prepared_dataset = prepare_training_dataset(raw_dataset, dataset_config)
    preprocessor = build_preprocessor(prepared_dataset.feature_spec)
    transformed = preprocessor.fit_transform(prepared_dataset.features, prepared_dataset.target)
    splits = split_dataset(
        prepared_dataset.features,
        prepared_dataset.target,
        SplitConfig(test_size=0.25, validation_size=0.25, random_state=42),
    )

    assert transformed.shape[0] == raw_dataset.shape[0]
    assert transformed.shape[1] > prepared_dataset.features.shape[1]
    assert splits.X_validation is not None
    assert splits.y_validation is not None
    assert (
        len(splits.X_train) + len(splits.X_validation) + len(splits.X_test)
        == len(prepared_dataset.features)
    )
