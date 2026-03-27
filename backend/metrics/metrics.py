"""Serve pre-computed evaluation metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from backend.configs.config import get_settings

logger = logging.getLogger(__name__)

METRICS_FILENAME = "evaluation_metrics.json"


class ModelMetrics(BaseModel):
    """Metrics for a single model."""

    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    mAP50: float | None = None
    mAP50_95: float | None = None
    dice_coefficient: float | None = None
    iou: float | None = None
    pixel_accuracy: float | None = None


class EvaluationMetrics(BaseModel):
    """All model evaluation metrics."""

    classifier: ModelMetrics
    detector: ModelMetrics
    segmentor: ModelMetrics


def get_metrics() -> EvaluationMetrics:
    """Load metrics from JSON file.

    Looks for ``evaluation_metrics.json`` in the model directory.
    Raises an HTTP 503 error if the file is not available.
    """
    settings = get_settings()
    metrics_path = settings.MODEL_DIR / METRICS_FILENAME

    if not metrics_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Evaluation metrics not available. "
                "Run the training pipeline first: python -m training"
            ),
        )

    try:
        data = json.loads(metrics_path.read_text())
        logger.info("Loaded evaluation metrics from %s", metrics_path)
        return EvaluationMetrics(
            classifier=ModelMetrics(**data.get("classifier", {})),
            detector=ModelMetrics(**data.get("detector", {})),
            segmentor=ModelMetrics(**data.get("segmentor", {})),
        )
    except Exception as exc:
        logger.error("Failed to parse %s: %s", metrics_path, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse evaluation metrics: {exc}",
        ) from exc
