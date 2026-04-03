"""Prometheus endpoint helpers for the FastAPI application."""

from __future__ import annotations

from prometheus_client import CollectorRegistry
from prometheus_client.asgi import make_asgi_app


def build_metrics_asgi_app(registry: CollectorRegistry):
    """Build the ASGI app that exposes Prometheus metrics."""

    return make_asgi_app(registry=registry)
