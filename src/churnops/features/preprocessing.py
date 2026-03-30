"""Preprocessing and split logic for churn model training."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnops.config import DatasetConfig, SplitConfig


@dataclass(slots=True)
class FeatureSpec:
    """Resolved numeric and categorical feature groups."""

    numeric_features: list[str]
    categorical_features: list[str]

    @property
    def all_features(self) -> list[str]:
        """Return all feature names in training order."""

        return [*self.numeric_features, *self.categorical_features]


@dataclass(slots=True)
class PreparedDataset:
    """Prepared feature frame, encoded target, and feature metadata."""

    features: pd.DataFrame
    target: pd.Series
    feature_spec: FeatureSpec


@dataclass(slots=True)
class DataSplits:
    """Train, validation, and test partitions for supervised learning."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame | None
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series | None
    y_test: pd.Series


def prepare_training_dataset(
    dataframe: pd.DataFrame,
    config: DatasetConfig,
) -> PreparedDataset:
    """Prepare raw churn data for model training."""

    working_frame = dataframe.copy()

    for column in config.numeric_coercion_columns:
        if column not in working_frame.columns:
            raise ValueError(f"Configured numeric coercion column '{column}' was not found.")
        working_frame[column] = pd.to_numeric(working_frame[column], errors="coerce")

    target = _encode_target(working_frame[config.target_column], config.positive_class)

    removable_columns = [config.target_column, *config.id_columns, *config.drop_columns]
    missing_removals = sorted(set(removable_columns).difference(working_frame.columns))
    if missing_removals:
        raise ValueError(
            "Configured feature exclusions were not found in the dataset: "
            + ", ".join(missing_removals)
        )

    features = working_frame.drop(columns=removable_columns)
    feature_spec = _resolve_feature_spec(features, config)
    return PreparedDataset(features=features, target=target, feature_spec=feature_spec)


def build_preprocessor(feature_spec: FeatureSpec) -> ColumnTransformer:
    """Build a sklearn preprocessing pipeline for mixed-type churn features."""

    if not feature_spec.all_features:
        raise ValueError("At least one training feature is required to build the preprocessor.")

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if feature_spec.numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, feature_spec.numeric_features))

    if feature_spec.categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, feature_spec.categorical_features)
        )

    return ColumnTransformer(transformers=transformers)


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    config: SplitConfig,
) -> DataSplits:
    """Create train, validation, and test splits with stratification when possible."""

    stratify_target = target if target.nunique() > 1 else None

    try:
        X_train_pool, X_test, y_train_pool, y_test = train_test_split(
            features,
            target,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify_target,
        )
    except ValueError as error:
        raise ValueError(
            "Unable to create the configured test split. Ensure each target class has enough rows."
        ) from error

    if config.validation_size == 0:
        return DataSplits(
            X_train=X_train_pool,
            X_validation=None,
            X_test=X_test,
            y_train=y_train_pool,
            y_validation=None,
            y_test=y_test,
        )

    validation_size = config.validation_size / (1 - config.test_size)
    stratify_pool = y_train_pool if y_train_pool.nunique() > 1 else None

    try:
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_pool,
            y_train_pool,
            test_size=validation_size,
            random_state=config.random_state,
            stratify=stratify_pool,
        )
    except ValueError as error:
        raise ValueError(
            "Unable to create the configured validation split. Increase the dataset size or "
            "reduce split.validation_size."
        ) from error

    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def _encode_target(target: pd.Series, positive_class: str) -> pd.Series:
    """Encode the churn target into binary labels."""

    if target.isna().any():
        raise ValueError("Target column contains missing values.")

    if is_bool_dtype(target):
        return target.astype(int)

    if is_numeric_dtype(target):
        unique_values = set(target.dropna().astype(int).unique().tolist())
        if unique_values.issubset({0, 1}):
            return target.astype(int)

    normalized_target = target.astype("string").str.strip()
    observed_classes = sorted(normalized_target.dropna().unique().tolist())
    if len(observed_classes) != 2:
        raise ValueError(
            "Binary churn training expects exactly two target classes, "
            f"but found {observed_classes}."
        )
    if positive_class not in observed_classes:
        raise ValueError(
            f"Configured positive class '{positive_class}' was not found in target values."
        )

    return normalized_target.eq(positive_class).astype(int)


def _resolve_feature_spec(
    features: pd.DataFrame,
    config: DatasetConfig,
) -> FeatureSpec:
    """Resolve and validate numeric and categorical feature assignments."""

    configured_numeric = _validate_configured_columns(
        config.numeric_features,
        features,
        "data.numeric_features",
    )
    configured_categorical = _validate_configured_columns(
        config.categorical_features,
        features,
        "data.categorical_features",
    )

    overlap = set(configured_numeric).intersection(configured_categorical)
    if overlap:
        raise ValueError(
            "Feature columns cannot be both numeric and categorical: "
            + ", ".join(sorted(overlap))
        )

    assigned_columns = set(configured_numeric).union(configured_categorical)
    remaining_columns = [column for column in features.columns if column not in assigned_columns]

    inferred_numeric = [
        column for column in remaining_columns if is_numeric_dtype(features[column])
    ]
    inferred_categorical = [
        column for column in remaining_columns if column not in inferred_numeric
    ]

    return FeatureSpec(
        numeric_features=[*configured_numeric, *inferred_numeric],
        categorical_features=[*configured_categorical, *inferred_categorical],
    )


def _validate_configured_columns(
    configured_columns: list[str],
    features: pd.DataFrame,
    section_name: str,
) -> list[str]:
    """Ensure configured feature names exist in the prepared feature frame."""

    missing_columns = sorted(set(configured_columns).difference(features.columns))
    if missing_columns:
        raise ValueError(
            f"Configured columns in '{section_name}' were not found: "
            + ", ".join(missing_columns)
        )

    return [column for column in configured_columns if column in features.columns]
