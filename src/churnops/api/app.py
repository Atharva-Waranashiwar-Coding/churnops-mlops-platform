"""FastAPI application bootstrap for churn inference."""

from __future__ import annotations

import logging
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churnops import __version__
from churnops.api.middleware import RequestContextMiddleware
from churnops.api.routes import router
from churnops.config import Settings, get_default_config_path, load_runtime_settings
from churnops.inference import InferenceService
from churnops.inference.exceptions import InferenceError, ModelLoadError, PredictionError
from churnops.monitoring import (
    InferenceMetrics,
    RequestMetricsMiddleware,
    build_metrics_asgi_app,
)

LOGGER = logging.getLogger(__name__)


def create_app(
    config_path: str | Path | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create the FastAPI application for churn inference."""

    resolved_settings = settings or load_runtime_settings(config_path)
    metrics = InferenceMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = InferenceService(resolved_settings, metrics=metrics)
        app.state.settings = resolved_settings
        app.state.inference_service = service
        LOGGER.info(
            "Starting inference service | version=%s config=%s model_source=%s preload_model=%s "
            "tracking_enabled=%s drift_enabled=%s",
            __version__,
            resolved_settings.config_path,
            resolved_settings.inference.model_source,
            resolved_settings.inference.preload_model,
            resolved_settings.tracking.enabled,
            resolved_settings.drift.enabled,
        )
        if resolved_settings.inference.preload_model:
            try:
                service.preload_model()
            except ModelLoadError:
                LOGGER.exception("Inference model preload failed.")
        yield

    app = FastAPI(
        title="ChurnOps Inference API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.metrics = metrics
    app.add_middleware(RequestMetricsMiddleware, metrics=metrics)
    app.add_middleware(RequestContextMiddleware)
    app.mount("/metrics", build_metrics_asgi_app(metrics.registry))
    _register_exception_handlers(app)
    app.include_router(router)
    return app


def build_argument_parser() -> ArgumentParser:
    """Build the CLI parser for local API execution."""

    parser = ArgumentParser(description="Run the ChurnOps inference API.")
    parser.add_argument(
        "--config",
        default=get_default_config_path(),
        help="Path to the YAML configuration file.",
    )
    return parser


def main() -> int:
    """Run the FastAPI application with the configured host and port."""

    parser = build_argument_parser()
    args = parser.parse_args()

    from churnops.runtime_logging import configure_logging

    configure_logging("churnops-inference-api")

    try:
        settings = load_runtime_settings(args.config)
    except FileNotFoundError as error:
        LOGGER.error("%s. Provide a valid config path with --config.", error)
        return 1
    except Exception:
        LOGGER.exception("Unable to load inference runtime settings.")
        return 1

    LOGGER.info(
        "Serving inference API | host=%s port=%s model_source=%s preload_model=%s config=%s",
        settings.inference.host,
        settings.inference.port,
        settings.inference.model_source,
        settings.inference.preload_model,
        settings.config_path,
    )

    try:
        uvicorn.run(
            create_app(settings=settings),
            host=settings.inference.host,
            port=settings.inference.port,
            log_level="info",
            access_log=False,
            log_config=None,
        )
    except Exception:
        LOGGER.exception("Inference API terminated unexpectedly.")
        return 1
    return 0


def _register_exception_handlers(app: FastAPI) -> None:
    """Register JSON exception handlers for service-level failures."""

    @app.exception_handler(ModelLoadError)
    async def handle_model_load_error(_: Request, error: ModelLoadError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.exception_handler(PredictionError)
    async def handle_prediction_error(_: Request, error: PredictionError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(error)})

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        _: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Request validation failed.",
                "errors": jsonable_encoder(error.errors()),
            },
        )

    @app.exception_handler(InferenceError)
    async def handle_inference_error(_: Request, error: InferenceError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(error)})


if __name__ == "__main__":
    raise SystemExit(main())
