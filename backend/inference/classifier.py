"""MobileNet-based dental caries classifier."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.configs.config import Settings, get_settings

logger = logging.getLogger(__name__)

LABELS = ["No Caries", "Caries"]


@dataclass(frozen=True)
class ClassificationResult:
    """Result of a classification inference."""

    label: str
    confidence: float
    probabilities: dict[str, float]


def _demo_classify() -> ClassificationResult:
    """Generate a realistic demo classification (70% caries, 30% no caries)."""
    is_caries = random.random() < 0.70
    if is_caries:
        confidence = round(random.uniform(0.72, 0.97), 4)
        return ClassificationResult(
            label="Caries",
            confidence=confidence,
            probabilities={
                "Caries": confidence,
                "No Caries": round(1.0 - confidence, 4),
            },
        )
    confidence = round(random.uniform(0.65, 0.92), 4)
    return ClassificationResult(
        label="No Caries",
        confidence=confidence,
        probabilities={
            "No Caries": confidence,
            "Caries": round(1.0 - confidence, 4),
        },
    )


def classify(
    preprocessed: NDArray[np.float32],
    model: Any | None = None,
    settings: Settings | None = None,
) -> ClassificationResult:
    """Run classification on a preprocessed image.

    Args:
        preprocessed: Image tensor of shape (1, 224, 224, 3), float32, [0,1].
        model: ONNX InferenceSession or None for demo mode.
        settings: Application settings.

    Returns:
        ClassificationResult with label, confidence, and per-class probabilities.
    """
    if settings is None:
        settings = get_settings()

    if model is None:
        logger.info("Classifier running in DEMO mode")
        return _demo_classify()

    # Production inference path
    input_name = model.get_inputs()[0].name
    output = model.run(None, {input_name: preprocessed})
    probs = output[0][0]

    idx = int(np.argmax(probs))
    label = LABELS[idx]
    confidence = float(probs[idx])

    result = ClassificationResult(
        label=label,
        confidence=round(confidence, 4),
        probabilities={
            LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))
        },
    )
    logger.info("Classification: %s (%.2f%%)", result.label, result.confidence * 100)
    return result
