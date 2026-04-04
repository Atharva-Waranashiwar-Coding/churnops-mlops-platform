"""Training-time reference baseline generation for drift monitoring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from churnops.config import DriftConfig
from churnops.drift.models import DriftBaseline, DriftFeatureBaseline
from churnops.features.preprocessing import FeatureSpec

_MISSING_BUCKET = "__missing__"
_OTHER_BUCKET = "__other__"


def build_drift_baseline(
    feature_frame: pd.DataFrame,
    feature_spec: FeatureSpec,
    config: DriftConfig,
) -> DriftBaseline:
    """Build a reference baseline from the model training split."""

    features: dict[str, DriftFeatureBaseline] = {}

    for feature_name in feature_spec.numeric_features:
        features[feature_name] = _build_numeric_feature_baseline(
            feature_name=feature_name,
            series=feature_frame[feature_name],
            bin_count=config.numeric_bin_count,
        )

    for feature_name in feature_spec.categorical_features:
        features[feature_name] = _build_categorical_feature_baseline(
            feature_name=feature_name,
            series=feature_frame[feature_name],
            top_k=config.categorical_top_k,
        )

    return DriftBaseline(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_split="train",
        sample_size=int(feature_frame.shape[0]),
        numeric_features=list(feature_spec.numeric_features),
        categorical_features=list(feature_spec.categorical_features),
        features=features,
    )


def load_drift_baseline(path: str | Path) -> DriftBaseline:
    """Load a persisted drift baseline from disk."""

    baseline_path = Path(path)
    with baseline_path.open("r", encoding="utf-8") as baseline_file:
        return DriftBaseline.from_payload(json.load(baseline_file))


def _build_numeric_feature_baseline(
    feature_name: str,
    series: pd.Series,
    bin_count: int,
) -> DriftFeatureBaseline:
    """Build a numeric reference distribution using quantile-derived PSI buckets."""

    numeric_series = pd.to_numeric(series, errors="coerce")
    cut_points = _resolve_numeric_cut_points(numeric_series, bin_count)
    bucket_labels, distribution = _calculate_numeric_distribution(numeric_series, cut_points)
    non_missing_series = numeric_series.dropna()

    return DriftFeatureBaseline(
        feature_name=feature_name,
        feature_type="numeric",
        sample_size=int(series.shape[0]),
        bucket_labels=bucket_labels,
        expected_distribution=distribution,
        cut_points=cut_points,
        metadata={
            "missing_rate": float(numeric_series.isna().mean()),
            "mean": float(non_missing_series.mean()) if not non_missing_series.empty else None,
            "std": (
                float(non_missing_series.std(ddof=0)) if not non_missing_series.empty else None
            ),
            "min": float(non_missing_series.min()) if not non_missing_series.empty else None,
            "max": float(non_missing_series.max()) if not non_missing_series.empty else None,
        },
    )


def _build_categorical_feature_baseline(
    feature_name: str,
    series: pd.Series,
    top_k: int,
) -> DriftFeatureBaseline:
    """Build a categorical reference distribution with explicit missing and other buckets."""

    normalized_series = _normalize_categorical_series(series)
    tracked_categories = [
        category
        for category in normalized_series[normalized_series != _MISSING_BUCKET]
        .value_counts()
        .head(top_k)
        .index.tolist()
    ]
    bucket_labels, distribution = _calculate_categorical_distribution(
        normalized_series,
        tracked_categories,
    )

    return DriftFeatureBaseline(
        feature_name=feature_name,
        feature_type="categorical",
        sample_size=int(series.shape[0]),
        bucket_labels=bucket_labels,
        expected_distribution=distribution,
        tracked_categories=tracked_categories,
        metadata={
            "missing_rate": float(normalized_series.eq(_MISSING_BUCKET).mean()),
            "unique_value_count": int(normalized_series.nunique(dropna=False)),
            "top_categories": tracked_categories,
        },
    )


def _resolve_numeric_cut_points(series: pd.Series, bin_count: int) -> list[float]:
    """Derive stable internal cut points from the baseline series."""

    non_missing_series = series.dropna()
    unique_value_count = int(non_missing_series.nunique())
    if unique_value_count <= 1:
        return []

    effective_bin_count = min(bin_count, unique_value_count)
    _, bin_edges = pd.qcut(
        non_missing_series,
        q=effective_bin_count,
        duplicates="drop",
        retbins=True,
    )
    return [float(edge) for edge in bin_edges[1:-1]]


def _calculate_numeric_distribution(
    series: pd.Series,
    cut_points: list[float],
) -> tuple[list[str], list[float]]:
    """Calculate missing-aware numeric bucket probabilities."""

    total_rows = int(series.shape[0])
    if total_rows == 0:
        raise ValueError("Cannot build a drift baseline from an empty feature series.")

    missing_probability = float(series.isna().sum() / total_rows)
    non_missing_series = series.dropna()

    if not cut_points:
        return [_MISSING_BUCKET, "all_values"], [missing_probability, 1.0 - missing_probability]

    cut = pd.cut(
        non_missing_series,
        bins=[float("-inf"), *cut_points, float("inf")],
        include_lowest=True,
    )
    categories = list(cut.cat.categories)
    counts = cut.value_counts(sort=False).reindex(categories, fill_value=0)
    probabilities = [missing_probability, *[float(count / total_rows) for count in counts]]
    bucket_labels = [_MISSING_BUCKET, *[str(category) for category in categories]]
    return bucket_labels, probabilities


def _calculate_categorical_distribution(
    normalized_series: pd.Series,
    tracked_categories: list[str],
) -> tuple[list[str], list[float]]:
    """Calculate categorical bucket probabilities with missing and other buckets."""

    total_rows = int(normalized_series.shape[0])
    if total_rows == 0:
        raise ValueError("Cannot build a drift baseline from an empty feature series.")

    bucket_labels = [_MISSING_BUCKET, *tracked_categories, _OTHER_BUCKET]
    probabilities: list[float] = []

    for bucket_label in bucket_labels:
        if bucket_label == _OTHER_BUCKET:
            count = int(
                (~normalized_series.isin([_MISSING_BUCKET, *tracked_categories])).sum()
            )
        else:
            count = int(normalized_series.eq(bucket_label).sum())
        probabilities.append(float(count / total_rows))

    return bucket_labels, probabilities


def _normalize_categorical_series(series: pd.Series) -> pd.Series:
    """Normalize categorical values into stable strings and a dedicated missing bucket."""

    normalized_series = series.astype("string").str.strip()
    normalized_series = normalized_series.fillna(_MISSING_BUCKET)
    normalized_series = normalized_series.replace("", _MISSING_BUCKET)
    return normalized_series
