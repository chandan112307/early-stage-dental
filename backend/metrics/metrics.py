"""Serve pre-computed evaluation metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from backend.configs.config import get_settings

logger = logging.getLogger(__name__)

METRICS_FILENAME = "evaluation_metrics.json"

# Hardcoded demo metrics used when no real evaluation file exists
_DEMO_METRICS: dict[str, dict[str, float]] = {
    "classifier": {
        "accuracy": 0.984,
        "precision": 0.961,
        "recall": 0.948,
        "f1_score": 0.954,
    },
    "detector": {
        "mAP50": 0.912,
        "mAP50_95": 0.743,
        "precision": 0.934,
        "recall": 0.891,
    },
    "segmentor": {
        "dice_coefficient": 0.887,
        "iou": 0.821,
        "pixel_accuracy": 0.964,
    },
}


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
    demo_mode: bool = True


def get_metrics() -> EvaluationMetrics:
    """Load metrics from JSON file or return demo metrics.

    Looks for ``evaluation_metrics.json`` in the model directory.
    """
    settings = get_settings()
    metrics_path = settings.MODEL_DIR / METRICS_FILENAME

    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text())
            logger.info("Loaded evaluation metrics from %s", metrics_path)
            return EvaluationMetrics(
                classifier=ModelMetrics(**data.get("classifier", {})),
                detector=ModelMetrics(**data.get("detector", {})),
                segmentor=ModelMetrics(**data.get("segmentor", {})),
                demo_mode=False,
            )
        except Exception:
            logger.warning(
                "Failed to parse %s – returning demo metrics",
                metrics_path,
                exc_info=True,
            )

    logger.info("Returning demo evaluation metrics")
    return EvaluationMetrics(
        classifier=ModelMetrics(**_DEMO_METRICS["classifier"]),
        detector=ModelMetrics(**_DEMO_METRICS["detector"]),
        segmentor=ModelMetrics(**_DEMO_METRICS["segmentor"]),
        demo_mode=True,
    )
