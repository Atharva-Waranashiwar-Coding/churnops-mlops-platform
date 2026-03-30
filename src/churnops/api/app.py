"""FastAPI application bootstrap for churn inference."""

from __future__ import annotations

from argparse import ArgumentParser
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from churnops import __version__
from churnops.api.routes import router
from churnops.config import Settings, load_settings
from churnops.inference import InferenceService
from churnops.inference.exceptions import InferenceError, ModelLoadError, PredictionError

LOGGER = logging.getLogger(__name__)


def create_app(config_path: str | Path = "configs/base.yaml") -> FastAPI:
    """Create the FastAPI application for churn inference."""

    settings = load_settings(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = InferenceService(settings)
        app.state.settings = settings
        app.state.inference_service = service
        if settings.inference.preload_model:
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
        default="configs/base.yaml",
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

    settings = load_settings(args.config)
    uvicorn.run(
        create_app(args.config),
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

    @app.exception_handler(InferenceError)
    async def handle_inference_error(_: Request, error: InferenceError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(error)})


if __name__ == "__main__":
    raise SystemExit(main())
