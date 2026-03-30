"""Tracker abstractions for the training pipeline."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

from churnops.config import Settings
from churnops.tracking.models import CompletedTrainingRun, TrackingResult


class TrainingTracker(Protocol):
    """Protocol implemented by concrete training run trackers."""

    def start_run(self) -> AbstractContextManager[None]:
        """Open the tracking context for a pipeline execution."""

    def finalize_run(self, completed_run: CompletedTrainingRun) -> TrackingResult:
        """Record training outputs after a successful pipeline execution."""


class NullTrainingTracker:
    """No-op tracker used when experiment tracking is disabled."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def start_run(self) -> AbstractContextManager[None]:
        """Return a no-op context manager."""

        return nullcontext()

    def finalize_run(self, completed_run: CompletedTrainingRun) -> TrackingResult:
        """Return a disabled tracking result without side effects."""

        del completed_run
        return TrackingResult(
            enabled=False,
            backend="none",
            tracking_uri=self._settings.tracking.tracking_uri,
            registry_uri=self._settings.tracking.registry_uri,
        )


def build_training_tracker(settings: Settings) -> TrainingTracker:
    """Create the configured training tracker implementation."""

    if not settings.tracking.enabled:
        return NullTrainingTracker(settings)

    from churnops.tracking.mlflow import MLflowTrainingTracker

    return MLflowTrainingTracker(settings)
