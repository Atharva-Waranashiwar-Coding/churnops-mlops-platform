"""HTTP middleware for request context and access logging."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from churnops.runtime_logging import reset_request_id, set_request_id

LOGGER = logging.getLogger("churnops.api.access")
_SKIP_LOG_PATHS = {"/metrics", "/health/live", "/health/ready"}


class RequestContextMiddleware:
    """Attach request IDs, response headers, and access logs to API traffic."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Wrap the downstream ASGI app with request-scoped logging context."""

        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = _resolve_request_id(scope)
        method = str(scope.get("method", "GET"))
        path = str(scope.get("path", "/"))
        status_code = 500
        start_time = perf_counter()
        token = set_request_id(request_id)

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        except Exception:
            if path not in _SKIP_LOG_PATHS:
                LOGGER.exception(
                    "Request failed | method=%s path=%s status_code=%s duration_ms=%.2f",
                    method,
                    path,
                    status_code,
                    (perf_counter() - start_time) * 1000,
                )
            raise
        else:
            if path not in _SKIP_LOG_PATHS:
                LOGGER.info(
                    "Request completed | method=%s path=%s status_code=%s duration_ms=%.2f",
                    method,
                    path,
                    status_code,
                    (perf_counter() - start_time) * 1000,
                )
        finally:
            reset_request_id(token)


def _resolve_request_id(scope: Scope) -> str:
    """Return the incoming request ID or generate a new one."""

    candidate = Headers(scope=scope).get("x-request-id", "").strip()
    if candidate:
        return candidate[:128]
    return uuid4().hex
