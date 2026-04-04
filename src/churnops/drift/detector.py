"""Feature distribution drift detection using population stability index."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from churnops.config import DriftConfig
from churnops.drift.baseline import (
    _calculate_categorical_distribution,
    _calculate_numeric_distribution,
    _normalize_categorical_series,
)
from churnops.drift.models import DriftBaseline, DriftEvaluationResult, FeatureDriftResult

_EPSILON = 1e-6


def evaluate_feature_distribution_drift(
    observed_frame: pd.DataFrame,
    baseline: DriftBaseline,
    config: DriftConfig,
) -> DriftEvaluationResult:
    """Compare live inference inputs against the training baseline using PSI."""

    observed_sample_size = int(observed_frame.shape[0])
    if observed_sample_size < config.min_samples:
        return DriftEvaluationResult(
            evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
            status="insufficient_data",
            observed_sample_size=observed_sample_size,
            min_samples=config.min_samples,
            window_size=config.window_size,
            warning_threshold=config.psi_warning_threshold,
            drift_threshold=config.psi_drift_threshold,
            min_drifted_features=config.min_drifted_features,
            drifted_feature_count=0,
            warning_feature_count=0,
            max_feature_psi=0.0,
            feature_results=[],
        )

    feature_results: list[FeatureDriftResult] = []
    for feature_name in [*baseline.numeric_features, *baseline.categorical_features]:
        feature_baseline = baseline.features[feature_name]
        observed_distribution = _calculate_observed_distribution(
            series=observed_frame.get(
                feature_name,
                pd.Series([pd.NA] * observed_sample_size),
            ),
            baseline=feature_baseline,
        )
        psi = calculate_population_stability_index(
            feature_baseline.expected_distribution,
            observed_distribution,
        )
        feature_results.append(
            FeatureDriftResult(
                feature_name=feature_name,
                feature_type=feature_baseline.feature_type,
                psi=psi,
                status=_resolve_feature_status(psi, config),
                bucket_labels=list(feature_baseline.bucket_labels),
                baseline_distribution=list(feature_baseline.expected_distribution),
                observed_distribution=observed_distribution,
            )
        )

    feature_results.sort(key=lambda result: result.psi, reverse=True)
    drifted_feature_count = sum(result.status == "drift" for result in feature_results)
    warning_feature_count = sum(result.status == "warning" for result in feature_results)
    max_feature_psi = max((result.psi for result in feature_results), default=0.0)
    status = _resolve_overall_status(
        drifted_feature_count=drifted_feature_count,
        warning_feature_count=warning_feature_count,
        config=config,
    )

    return DriftEvaluationResult(
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        status=status,
        observed_sample_size=observed_sample_size,
        min_samples=config.min_samples,
        window_size=config.window_size,
        warning_threshold=config.psi_warning_threshold,
        drift_threshold=config.psi_drift_threshold,
        min_drifted_features=config.min_drifted_features,
        drifted_feature_count=drifted_feature_count,
        warning_feature_count=warning_feature_count,
        max_feature_psi=max_feature_psi,
        feature_results=feature_results,
    )


def calculate_population_stability_index(
    expected_distribution: list[float],
    observed_distribution: list[float],
) -> float:
    """Calculate PSI over two aligned discrete distributions."""

    if len(expected_distribution) != len(observed_distribution):
        raise ValueError(
            "Expected and observed distributions must contain the same number of bins."
        )

    psi = 0.0
    for expected_probability, observed_probability in zip(
        expected_distribution,
        observed_distribution,
        strict=True,
    ):
        stabilized_expected = max(float(expected_probability), _EPSILON)
        stabilized_observed = max(float(observed_probability), _EPSILON)
        psi += (stabilized_observed - stabilized_expected) * (
            math.log(stabilized_observed / stabilized_expected)
        )
    return float(psi)


def _calculate_observed_distribution(
    series: pd.Series,
    baseline,
) -> list[float]:
    """Calculate the live distribution for a feature using the baseline bucket definition."""

    if baseline.feature_type == "numeric":
        _, observed_distribution = _calculate_numeric_distribution(
            pd.to_numeric(series, errors="coerce"),
            baseline.cut_points,
        )
        return observed_distribution

    _, observed_distribution = _calculate_categorical_distribution(
        _normalize_categorical_series(series),
        baseline.tracked_categories,
    )
    return observed_distribution


def _resolve_feature_status(psi: float, config: DriftConfig) -> str:
    """Map a PSI value onto a stable feature-level drift status."""

    if psi >= config.psi_drift_threshold:
        return "drift"
    if psi >= config.psi_warning_threshold:
        return "warning"
    return "stable"


def _resolve_overall_status(
    drifted_feature_count: int,
    warning_feature_count: int,
    config: DriftConfig,
) -> str:
    """Map feature-level results onto an overall drift decision."""

    if drifted_feature_count >= config.min_drifted_features:
        return "drift_detected"
    if drifted_feature_count > 0 or warning_feature_count > 0:
        return "warning"
    return "stable"
