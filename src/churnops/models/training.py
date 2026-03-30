"""Model training orchestration for the churn baseline classifier."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churnops.config import ModelConfig
from churnops.features.preprocessing import DataSplits, FeatureSpec, build_preprocessor
from churnops.models.evaluation import evaluate_classifier


@dataclass(slots=True)
class TrainingResult:
    """In-memory result of a completed training run."""

    model_pipeline: Pipeline
    metrics: dict[str, dict[str, float | int | None]]
    split_sizes: dict[str, int]
    feature_spec: FeatureSpec


def train_baseline_model(
    data_splits: DataSplits,
    feature_spec: FeatureSpec,
    config: ModelConfig,
) -> TrainingResult:
    """Train the configured baseline classifier and evaluate all available splits."""

    estimator = build_estimator(config)
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_spec)),
            ("classifier", estimator),
        ]
    )
    model_pipeline.fit(data_splits.X_train, data_splits.y_train)

    metrics: dict[str, dict[str, float | int | None]] = {
        "train": evaluate_classifier(model_pipeline, data_splits.X_train, data_splits.y_train),
        "test": evaluate_classifier(model_pipeline, data_splits.X_test, data_splits.y_test),
    }

    split_sizes = {
        "train": int(data_splits.y_train.shape[0]),
        "test": int(data_splits.y_test.shape[0]),
    }

    if data_splits.X_validation is not None and data_splits.y_validation is not None:
        metrics["validation"] = evaluate_classifier(
            model_pipeline,
            data_splits.X_validation,
            data_splits.y_validation,
        )
        split_sizes["validation"] = int(data_splits.y_validation.shape[0])

    return TrainingResult(
        model_pipeline=model_pipeline,
        metrics=metrics,
        split_sizes=split_sizes,
        feature_spec=feature_spec,
    )


def build_estimator(config: ModelConfig) -> LogisticRegression:
    """Build the configured baseline estimator."""

    if config.name != "logistic_regression":
        raise ValueError(
            "Phase 01 supports only 'logistic_regression' as the model.name value."
        )

    return LogisticRegression(**config.params)
