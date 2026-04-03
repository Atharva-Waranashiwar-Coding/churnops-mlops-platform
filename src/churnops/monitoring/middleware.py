"""ASGI middleware for request-level metrics collection."""

from __future__ import annotations

from time import perf_counter

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from churnops.monitoring.metrics import InferenceMetrics


class RequestMetricsMiddleware:
    """Collect request-count, failure, and latency metrics for HTTP traffic."""

    def __init__(self, app: ASGIApp, metrics: InferenceMetrics) -> None:
        self._app = app
        self._metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Wrap the downstream ASGI app and record request metrics."""

        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_path = scope.get("path", "")
        if request_path.startswith("/metrics"):
            await self._app(scope, receive, send)
            return

        status_code = 500
        start_time = perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        except Exception:
            self._metrics.record_http_request(
                method=str(scope.get("method", "GET")),
                route=_resolve_route_label(scope),
                status_code=status_code,
                duration_seconds=perf_counter() - start_time,
            )
            raise

        self._metrics.record_http_request(
            method=str(scope.get("method", "GET")),
            route=_resolve_route_label(scope),
            status_code=status_code,
            duration_seconds=perf_counter() - start_time,
        )


def _resolve_route_label(scope: Scope) -> str:
    """Return a low-cardinality route label for metrics."""

    route = scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    return "/unmatched"
