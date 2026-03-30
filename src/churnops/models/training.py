"""Model training orchestration for the churn baseline classifier."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churnops.config import ModelConfig
from churnops.features.preprocessing import FeatureSpec, build_preprocessor


@dataclass(slots=True)
class TrainedModel:
    """Trained sklearn pipeline and its feature metadata."""

    model_pipeline: Pipeline
    feature_spec: FeatureSpec


def train_baseline_model(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_spec: FeatureSpec,
    config: ModelConfig,
) -> TrainedModel:
    """Train the configured baseline classifier."""

    estimator = build_estimator(config)
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_spec)),
            ("classifier", estimator),
        ]
    )
    model_pipeline.fit(train_features, train_target)

    return TrainedModel(
        model_pipeline=model_pipeline,
        feature_spec=feature_spec,
    )


def build_estimator(config: ModelConfig) -> LogisticRegression:
    """Build the configured baseline estimator."""

    if config.name != "logistic_regression":
        raise ValueError(
            "Phase 02 supports only 'logistic_regression' as the model.name value."
        )

    return LogisticRegression(**config.params)
