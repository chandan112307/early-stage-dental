"""API routes for the dental caries detection backend."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from backend.configs.config import get_settings
from backend.inference.pipeline import PipelineResult, run_pipeline
from backend.metrics.metrics import EvaluationMetrics, get_metrics
from backend.model_loader.loader import get_models
from backend.postprocessing.postprocess import postprocess
from backend.preprocessing.preprocess import (
    PreprocessingError,
    load_image,
    validate_file,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class BoundingBoxSchema(BaseModel):
    """Single detection bounding box."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str
    confidence: float


class SegmentationData(BaseModel):
    """Segmentation result summary."""

    affected_percentage: float
    mask_base64: str


class ModelInfo(BaseModel):
    """Runtime information about the models."""

    classifier_loaded: bool
    detector_loaded: bool
    segmentor_loaded: bool


class PredictionResponse(BaseModel):
    """Full response from the /predict endpoint."""

    prediction: str
    confidence: float
    probabilities: dict[str, float]
    processed_image_url: str
    annotated_base64: str
    bounding_boxes: list[BoundingBoxSchema]
    segmentation_data: SegmentationData | None
    model_info: ModelInfo
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str
    models_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile) -> PredictionResponse:
    """Accept an uploaded dental image and return caries predictions.

    The pipeline runs classification first; if caries are detected it
    additionally runs detection and segmentation.
    """
    settings = get_settings()

    # --- Validate upload ---
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    contents = await file.read()
    try:
        validate_file(file.filename, len(contents), settings)
    except PreprocessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Decode image ---
    try:
        image = load_image(contents)
    except PreprocessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Run inference pipeline ---
    start = time.perf_counter()
    try:
        result: PipelineResult = run_pipeline(image, settings=settings)
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(
            status_code=500, detail="Inference pipeline error"
        ) from exc
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    # --- Postprocess ---
    post = postprocess(
        image,
        result.bounding_boxes,
        result.segmentation,
        settings=settings,
    )

    # --- Build response ---
    models = get_models()
    seg_data: SegmentationData | None = None
    if result.segmentation is not None:
        seg_data = SegmentationData(
            affected_percentage=result.segmentation.affected_percentage,
            mask_base64=post["mask_base64"],
        )

    return PredictionResponse(
        prediction=result.classification.label,
        confidence=result.classification.confidence,
        probabilities=result.classification.probabilities,
        processed_image_url=post["processed_image_url"],
        annotated_base64=post["annotated_base64"],
        bounding_boxes=[
            BoundingBoxSchema(
                x_min=b.x_min,
                y_min=b.y_min,
                x_max=b.x_max,
                y_max=b.y_max,
                label=b.label,
                confidence=b.confidence,
            )
            for b in result.bounding_boxes
        ],
        segmentation_data=seg_data,
        model_info=ModelInfo(
            classifier_loaded=models.classifier is not None,
            detector_loaded=models.detector is not None,
            segmentor_loaded=models.segmentor is not None,
        ),
        inference_time_ms=elapsed_ms,
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Lightweight health-check endpoint."""
    models = get_models()
    return HealthResponse(
        status="ok",
        models_loaded=all(
            m is not None
            for m in [models.classifier, models.detector, models.segmentor]
        ),
    )


@router.get("/metrics", response_model=EvaluationMetrics)
async def metrics() -> EvaluationMetrics:
    """Return pre-computed model evaluation metrics."""
    return get_metrics()
