"""U-Net-based dental caries segmentor."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.configs.config import Settings, get_settings
from backend.inference.detector import BoundingBox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentationResult:
    """Result of segmentation inference."""

    mask: NDArray[np.uint8]
    affected_percentage: float


def _demo_segment(
    image_width: int,
    image_height: int,
    boxes: list[BoundingBox],
) -> SegmentationResult:
    """Generate a plausible mock segmentation mask based on detected boxes."""
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for box in boxes:
        # Create an elliptical region inside each bounding box
        cx = (box.x_min + box.x_max) // 2
        cy = (box.y_min + box.y_max) // 2
        rx = max(1, (box.x_max - box.x_min) // 2 - random.randint(2, 6))
        ry = max(1, (box.y_max - box.y_min) // 2 - random.randint(2, 6))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    total_pixels = image_width * image_height
    affected = int(np.count_nonzero(mask))
    pct = round((affected / total_pixels) * 100, 2) if total_pixels > 0 else 0.0

    logger.info("Demo segmentor: %.2f%% affected area", pct)
    return SegmentationResult(mask=mask, affected_percentage=pct)


def segment(
    preprocessed: NDArray[np.float32],
    original_width: int,
    original_height: int,
    boxes: list[BoundingBox],
    model: Any | None = None,
    settings: Settings | None = None,
) -> SegmentationResult:
    """Run segmentation on a preprocessed image.

    Args:
        preprocessed: Image tensor of shape (1, 256, 256, 3), float32, [0,1].
        original_width: Width of the original image.
        original_height: Height of the original image.
        boxes: Detected bounding boxes (used by demo mode for plausible masks).
        model: ONNX InferenceSession or None for demo mode.
        settings: Application settings.

    Returns:
        SegmentationResult with binary mask and affected area percentage.
    """
    if settings is None:
        settings = get_settings()

    if model is None:
        logger.info("Segmentor running in DEMO mode")
        return _demo_segment(original_width, original_height, boxes)

    # Production inference path
    input_name = model.get_inputs()[0].name
    output = model.run(None, {input_name: preprocessed})
    raw_mask = output[0][0]  # Expected shape: (256, 256) or (256, 256, 1)

    if raw_mask.ndim == 3:
        raw_mask = raw_mask[:, :, 0]

    binary = (raw_mask > 0.5).astype(np.uint8) * 255
    resized_mask: NDArray[np.uint8] = cv2.resize(
        binary,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    total_pixels = original_width * original_height
    affected = int(np.count_nonzero(resized_mask))
    pct = round((affected / total_pixels) * 100, 2) if total_pixels > 0 else 0.0

    logger.info("Segmentor: %.2f%% affected area", pct)
    return SegmentationResult(mask=resized_mask, affected_percentage=pct)
