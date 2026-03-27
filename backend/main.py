"""FastAPI application entry point for the dental caries detection backend."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.configs.config import get_settings
from backend.model_loader.loader import ModelLoadError, load_models

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: load models once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle handler."""
    settings = get_settings()
    logger.info("Starting dental caries detection backend")

    try:
        load_models(settings)
    except ModelLoadError as exc:
        logger.error(
            "Model loading failed: %s. "
            "Run 'python -m training' to train and deploy models first.",
            exc,
        )
        raise SystemExit(1) from exc

    yield  # application runs

    logger.info("Shutting down backend")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Dental Caries Detection API",
        description=(
            "AI-powered backend for early-stage dental caries detection. "
            "Accepts dental radiograph images and returns classification, "
            "detection, and segmentation results."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files (processed output images)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/outputs",
        StaticFiles(directory=str(settings.OUTPUT_DIR)),
        name="outputs",
    )

    # API routes
    app.include_router(router, prefix="/api")

    return app


app = create_app()
