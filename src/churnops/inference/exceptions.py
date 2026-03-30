"""Inference service exceptions."""

from __future__ import annotations


class InferenceError(Exception):
    """Base exception for inference-service failures."""


class ModelConfigurationError(InferenceError):
    """Raised when the configured inference model source is invalid."""


class ModelLoadError(InferenceError):
    """Raised when the configured model cannot be loaded."""


class PredictionError(InferenceError):
    """Raised when prediction execution fails."""
