"""Inference pipeline orchestrator.

Flow: classify → if caries detected → detect + segment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from backend.configs.config import Settings, get_settings
from backend.inference.classifier import ClassificationResult, classify
from backend.inference.detector import BoundingBox, detect
from backend.inference.segmentor import SegmentationResult, segment
from backend.model_loader.loader import LoadedModels, get_models
from backend.preprocessing.preprocess import preprocess_for_model

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Aggregated result from the full inference pipeline."""

    classification: ClassificationResult
    bounding_boxes: list[BoundingBox]
    segmentation: SegmentationResult | None


def run_pipeline(
    image: NDArray[np.uint8],
    settings: Settings | None = None,
    models: LoadedModels | None = None,
) -> PipelineResult:
    """Execute the full inference pipeline on a raw BGR image.

    Steps:
        1. Preprocess for MobileNet → classify.
        2. If caries detected, preprocess for YOLO → detect.
        3. If caries detected, preprocess for U-Net → segment.

    Args:
        image: Raw BGR uint8 image from the uploaded file.
        settings: Application settings.
        models: Pre-loaded model objects.

    Returns:
        PipelineResult with classification, boxes, and segmentation.
    """
    if settings is None:
        settings = get_settings()
    if models is None:
        models = get_models()

    h, w = image.shape[:2]

    # Step 1: Classification
    cls_input = preprocess_for_model(image, settings.MOBILENET_SIZE)
    classification = classify(cls_input, model=models.classifier, settings=settings)

    bounding_boxes: list[BoundingBox] = []
    segmentation: SegmentationResult | None = None

    # Steps 2 & 3: Detection + Segmentation (only when caries found)
    if classification.label == "Caries":
        det_input = preprocess_for_model(image, settings.YOLO_SIZE)
        bounding_boxes = detect(
            det_input,
            original_width=w,
            original_height=h,
            model=models.detector,
            settings=settings,
        )

        seg_input = preprocess_for_model(image, settings.UNET_SIZE)
        segmentation = segment(
            seg_input,
            original_width=w,
            original_height=h,
            model=models.segmentor,
            settings=settings,
        )
    else:
        logger.info("No caries detected – skipping detection and segmentation")

    return PipelineResult(
        classification=classification,
        bounding_boxes=bounding_boxes,
        segmentation=segmentation,
    )
