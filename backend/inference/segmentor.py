"""U-Net-based dental caries segmentor."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.configs.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentationResult:
    """Result of segmentation inference."""

    mask: NDArray[np.uint8]
    affected_percentage: float


def segment(
    preprocessed: NDArray[np.float32],
    original_width: int,
    original_height: int,
    model: Any,
    settings: Settings | None = None,
) -> SegmentationResult:
    """Run segmentation on a preprocessed image.

    Args:
        preprocessed: Image tensor of shape (1, 256, 256, 3), float32, [0,1].
        original_width: Width of the original image.
        original_height: Height of the original image.
        model: ONNX InferenceSession (required).
        settings: Application settings.

    Returns:
        SegmentationResult with binary mask and affected area percentage.
    """
    if settings is None:
        settings = get_settings()

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
