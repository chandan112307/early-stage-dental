"""YOLO-based dental caries detector."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.configs.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundingBox:
    """A single detected region."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str
    confidence: float


def _demo_detect(image_width: int, image_height: int) -> list[BoundingBox]:
    """Generate 1-3 realistic bounding boxes at plausible dental locations."""
    num_boxes = random.randint(1, 3)
    boxes: list[BoundingBox] = []

    # Plausible regions within a dental radiograph (central area)
    for _ in range(num_boxes):
        cx = random.randint(int(image_width * 0.2), int(image_width * 0.8))
        cy = random.randint(int(image_height * 0.25), int(image_height * 0.75))
        half_w = random.randint(int(image_width * 0.04), int(image_width * 0.10))
        half_h = random.randint(int(image_height * 0.04), int(image_height * 0.10))

        boxes.append(
            BoundingBox(
                x_min=max(0, cx - half_w),
                y_min=max(0, cy - half_h),
                x_max=min(image_width, cx + half_w),
                y_max=min(image_height, cy + half_h),
                label="Caries",
                confidence=round(random.uniform(0.65, 0.95), 4),
            )
        )

    logger.info("Demo detector produced %d boxes", len(boxes))
    return boxes


def detect(
    preprocessed: NDArray[np.float32],
    original_width: int,
    original_height: int,
    model: Any | None = None,
    settings: Settings | None = None,
    confidence_threshold: float = 0.5,
) -> list[BoundingBox]:
    """Run object detection on a preprocessed image.

    Args:
        preprocessed: Image tensor of shape (1, 640, 640, 3), float32, [0,1].
        original_width: Width of the original (un-resized) image.
        original_height: Height of the original (un-resized) image.
        model: ONNX InferenceSession or None for demo mode.
        settings: Application settings.
        confidence_threshold: Minimum confidence to keep a detection.

    Returns:
        List of BoundingBox results in original image coordinates.
    """
    if settings is None:
        settings = get_settings()

    if model is None:
        logger.info("Detector running in DEMO mode")
        return _demo_detect(original_width, original_height)

    # Production inference path
    input_name = model.get_inputs()[0].name
    output = model.run(None, {input_name: preprocessed})
    detections = output[0]  # Expected shape: (1, N, 6) → [x1, y1, x2, y2, conf, cls]

    yolo_w, yolo_h = settings.YOLO_SIZE
    scale_x = original_width / yolo_w
    scale_y = original_height / yolo_h

    boxes: list[BoundingBox] = []
    for det in detections[0]:
        conf = float(det[4])
        if conf < confidence_threshold:
            continue
        boxes.append(
            BoundingBox(
                x_min=int(det[0] * scale_x),
                y_min=int(det[1] * scale_y),
                x_max=int(det[2] * scale_x),
                y_max=int(det[3] * scale_y),
                label="Caries",
                confidence=round(conf, 4),
            )
        )

    logger.info("Detector found %d boxes above threshold %.2f", len(boxes), confidence_threshold)
    return boxes
