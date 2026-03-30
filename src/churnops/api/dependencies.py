"""FastAPI dependency helpers for the inference service."""

from __future__ import annotations

from fastapi import Request

from churnops.inference import InferenceService


def get_inference_service(request: Request) -> InferenceService:
    """Return the application-scoped inference service."""

    return request.app.state.inference_service
