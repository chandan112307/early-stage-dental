"""Image preprocessing for dental caries inference pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.configs.config import Settings, get_settings

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when image preprocessing fails."""


def validate_file(filename: str, size_bytes: int, settings: Settings | None = None) -> None:
    """Validate uploaded file type and size.

    Raises:
        PreprocessingError: If the file type is unsupported or exceeds max size.
    """
    if settings is None:
        settings = get_settings()

    ext = Path(filename).suffix.lower()
    if ext not in settings.SUPPORTED_EXTENSIONS:
        raise PreprocessingError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(settings.SUPPORTED_EXTENSIONS))}"
        )

    if size_bytes > settings.MAX_FILE_SIZE_BYTES:
        raise PreprocessingError(
            f"File size {size_bytes / (1024 * 1024):.1f}MB exceeds "
            f"maximum {settings.MAX_FILE_SIZE_MB}MB"
        )


def load_image(file_bytes: bytes) -> NDArray[np.uint8]:
    """Decode raw bytes into a BGR numpy array.

    Raises:
        PreprocessingError: If the image cannot be decoded.
    """
    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise PreprocessingError("Could not decode image – file may be corrupt")
    logger.debug("Loaded image with shape %s", img.shape)
    return img


def apply_clahe(
    image: NDArray[np.uint8],
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> NDArray[np.uint8]:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances local contrast, which helps reveal subtle caries boundaries.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)

    merged = cv2.merge([enhanced_l, a_channel, b_channel])
    result: NDArray[np.uint8] = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    logger.debug("CLAHE applied (clip=%.1f, grid=%s)", clip_limit, tile_grid_size)
    return result


def resize_and_normalize(
    image: NDArray[np.uint8],
    target_size: tuple[int, int],
) -> NDArray[np.float32]:
    """Resize image and normalize pixel values to [0, 1].

    Args:
        image: BGR uint8 image.
        target_size: (width, height) for the model.

    Returns:
        Float32 array of shape (1, H, W, 3) with values in [0, 1].
    """
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    batched: NDArray[np.float32] = np.expand_dims(normalized, axis=0)
    return batched


def preprocess_for_model(
    image: NDArray[np.uint8],
    target_size: tuple[int, int],
) -> NDArray[np.float32]:
    """Full preprocessing: CLAHE → resize → normalize.

    This is the standard pipeline applied before every model inference.
    """
    enhanced = apply_clahe(image)
    return resize_and_normalize(enhanced, target_size)
