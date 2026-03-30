"""FastAPI application bootstrap for churn inference."""

from __future__ import annotations

from argparse import ArgumentParser
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn

from churnops import __version__
from churnops.api.routes import router
from churnops.config import Settings, get_default_config_path, load_runtime_settings
from churnops.inference import InferenceService
from churnops.inference.exceptions import InferenceError, ModelLoadError, PredictionError

LOGGER = logging.getLogger(__name__)


def create_app(
    config_path: str | Path | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create the FastAPI application for churn inference."""

    resolved_settings = settings or load_runtime_settings(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = InferenceService(resolved_settings)
        app.state.settings = resolved_settings
        app.state.inference_service = service
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    settings = load_runtime_settings(args.config)
    uvicorn.run(
        create_app(settings=settings),
        host=settings.inference.host,
        port=settings.inference.port,
        log_level="info",
    )
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
