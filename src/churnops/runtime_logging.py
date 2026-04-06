"""Runtime logging helpers for CLI and service entrypoints."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from contextvars import ContextVar, Token
from datetime import datetime, timezone

_REQUEST_ID: ContextVar[str | None] = ContextVar("churnops_request_id", default=None)
_VALID_LOG_FORMATS = {"text", "json"}


class RequestContextFilter(logging.Filter):
    """Inject request-scoped metadata into log records."""

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach service and request context to the emitted log record."""

        record.service = self._service_name
        record.request_id = get_request_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Render application logs as structured JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the log record into a stable JSON payload."""

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": getattr(record, "service", None),
            "request_id": getattr(record, "request_id", None),
            "message": record.getMessage(),
        }
        if payload["request_id"] == "-":
            payload["request_id"] = None
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            {key: value for key, value in payload.items() if value is not None},
            sort_keys=True,
        )


def configure_logging(
    service_name: str,
    env: Mapping[str, str] | None = None,
) -> None:
    """Configure process-wide logging using environment-driven settings."""

    environment = os.environ if env is None else env
    log_level = _resolve_log_level(environment.get("CHURNOPS_LOG_LEVEL", "INFO"))
    log_format = environment.get("CHURNOPS_LOG_FORMAT", "text").strip().lower() or "text"
    if log_format not in _VALID_LOG_FORMATS:
        log_format = "text"

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestContextFilter(service_name))
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(service)s | "
                "request_id=%(request_id)s | %(message)s"
            )
        )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)
    logging.captureWarnings(True)


def set_request_id(request_id: str) -> Token[str | None]:
    """Store the current request identifier in context-local state."""

    return _REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the previous request identifier."""

    _REQUEST_ID.reset(token)


def get_request_id() -> str | None:
    """Return the current request identifier, if one is set."""

    return _REQUEST_ID.get()


def _resolve_log_level(raw_level: str) -> int:
    """Map a string log level onto a stdlib logging constant."""

    resolved_level = logging.getLevelName(raw_level.strip().upper())
    return resolved_level if isinstance(resolved_level, int) else logging.INFO
