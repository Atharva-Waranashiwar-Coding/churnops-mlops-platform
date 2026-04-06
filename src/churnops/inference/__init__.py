"""Inference service interfaces for ChurnOps."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from churnops.inference.service import InferenceService

__all__ = ["InferenceService"]


def __getattr__(name: str):
    """Resolve the inference service lazily to avoid package import cycles."""

    if name == "InferenceService":
        from churnops.inference.service import InferenceService

        return InferenceService

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
