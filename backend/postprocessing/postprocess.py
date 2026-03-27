"""Postprocessing: draw bounding boxes, overlay segmentation, encode for API."""

from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.configs.config import Settings, get_settings
from backend.inference.detector import BoundingBox
from backend.inference.segmentor import SegmentationResult

logger = logging.getLogger(__name__)

# Teal/cyan colour for bounding boxes (BGR)
BBOX_COLOR = (200, 180, 0)
BBOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1

# Red overlay for segmentation mask
MASK_COLOR = (0, 0, 200)  # red in BGR
MASK_ALPHA = 0.35


def draw_bounding_boxes(
    image: NDArray[np.uint8],
    boxes: list[BoundingBox],
) -> NDArray[np.uint8]:
    """Draw teal bounding boxes with labels on a copy of the image."""
    canvas = image.copy()
    for box in boxes:
        cv2.rectangle(
            canvas,
            (box.x_min, box.y_min),
            (box.x_max, box.y_max),
            BBOX_COLOR,
            BBOX_THICKNESS,
        )
        label_text = f"{box.label} {box.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(
            canvas,
            (box.x_min, box.y_min - th - 6),
            (box.x_min + tw + 4, box.y_min),
            BBOX_COLOR,
            -1,
        )
        cv2.putText(
            canvas,
            label_text,
            (box.x_min + 2, box.y_min - 4),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return canvas


def overlay_segmentation(
    image: NDArray[np.uint8],
    seg_result: SegmentationResult | None,
) -> NDArray[np.uint8]:
    """Overlay a red semi-transparent segmentation mask onto the image."""
    if seg_result is None or not np.any(seg_result.mask):
        return image

    canvas = image.copy()
    overlay = canvas.copy()

    mask_bool = seg_result.mask > 0
    overlay[mask_bool] = MASK_COLOR

    cv2.addWeighted(overlay, MASK_ALPHA, canvas, 1 - MASK_ALPHA, 0, canvas)
    return canvas


def save_processed_image(
    image: NDArray[np.uint8],
    output_dir: Path,
) -> str:
    """Save the processed image and return its filename."""
    filename = f"result_{uuid.uuid4().hex[:12]}.png"
    out_path = output_dir / filename
    cv2.imwrite(str(out_path), image)
    logger.info("Saved processed image to %s", out_path)
    return filename


def encode_image_base64(image: NDArray[np.uint8]) -> str:
    """Encode a BGR image as a base64-encoded PNG string."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buffer).decode("utf-8")


def postprocess(
    original_image: NDArray[np.uint8],
    boxes: list[BoundingBox],
    seg_result: SegmentationResult | None,
    settings: Settings | None = None,
) -> dict[str, str]:
    """Run full postprocessing and return image URLs and base64 data.

    Returns:
        Dictionary with keys:
        - processed_image_url: relative URL to the saved annotated image
        - annotated_base64: base64-encoded annotated image
        - mask_base64: base64-encoded segmentation mask (or empty string)
    """
    if settings is None:
        settings = get_settings()

    # Draw annotations
    annotated = draw_bounding_boxes(original_image, boxes)
    annotated = overlay_segmentation(annotated, seg_result)

    # Save to disk
    filename = save_processed_image(annotated, settings.OUTPUT_DIR)

    # Encode to base64
    annotated_b64 = encode_image_base64(annotated)
    mask_b64 = ""
    if seg_result is not None and np.any(seg_result.mask):
        mask_bgr = cv2.cvtColor(seg_result.mask, cv2.COLOR_GRAY2BGR)
        mask_b64 = encode_image_base64(mask_bgr)

    return {
        "processed_image_url": f"/outputs/{filename}",
        "annotated_base64": annotated_b64,
        "mask_base64": mask_b64,
    }
