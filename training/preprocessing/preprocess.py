"""Image preprocessing utilities for dental X-ray images.

Provides functions for:
- CLAHE (Contrast Limited Adaptive Histogram Equalisation) enhancement
- Resizing to model-specific dimensions
- Pixel-value normalisation

All functions accept and return NumPy arrays so they integrate easily with
both TensorFlow and OpenCV pipelines.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from training.configs.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    MOBILENET_IMG_SIZE,
    UNET_IMG_SIZE,
    YOLO_IMG_SIZE,
)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """Apply CLAHE to enhance contrast in a dental X-ray image.

    Parameters
    ----------
    image:
        Input image as a NumPy array (grayscale or BGR).
    clip_limit:
        Threshold for contrast limiting.
    tile_grid_size:
        Size of the grid for histogram equalisation.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image in the same colour space as the input.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:
        # Grayscale
        return clahe.apply(image)

    if image.shape[2] == 1:
        return clahe.apply(image[:, :, 0])[:, :, np.newaxis]

    # For colour images, convert to LAB and apply CLAHE to the L channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize an image to ``target_size`` (width, height).

    Parameters
    ----------
    image:
        Input image.
    target_size:
        ``(width, height)`` tuple.
    interpolation:
        OpenCV interpolation flag.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalise pixel values to the ``[0, 1]`` range.

    Parameters
    ----------
    image:
        Input image with ``uint8`` or ``float`` values.

    Returns
    -------
    np.ndarray
        Image with ``float32`` values in ``[0, 1]``.
    """
    return image.astype(np.float32) / 255.0


def preprocess_for_mobilenet(
    image: np.ndarray,
    apply_clahe_enhancement: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline for MobileNet classification.

    Steps: CLAHE → resize to 224×224 → normalise to [0, 1].
    """
    if apply_clahe_enhancement:
        image = apply_clahe(image)
    image = resize_image(image, MOBILENET_IMG_SIZE)
    return normalize_image(image)


def preprocess_for_yolo(
    image: np.ndarray,
    apply_clahe_enhancement: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline for YOLO detection.

    Steps: CLAHE → resize to 640×640 → normalise to [0, 1].
    """
    if apply_clahe_enhancement:
        image = apply_clahe(image)
    image = resize_image(image, YOLO_IMG_SIZE)
    return normalize_image(image)


def preprocess_for_unet(
    image: np.ndarray,
    apply_clahe_enhancement: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline for U-Net segmentation.

    Steps: CLAHE → resize to 256×256 → normalise to [0, 1].
    """
    if apply_clahe_enhancement:
        image = apply_clahe(image)
    image = resize_image(image, UNET_IMG_SIZE)
    return normalize_image(image)


def load_and_preprocess(
    path: str,
    target_size: Tuple[int, int],
    grayscale: bool = False,
    apply_clahe_enhancement: bool = True,
) -> Optional[np.ndarray]:
    """Load an image from disk, apply CLAHE, resize, and normalise.

    Parameters
    ----------
    path:
        File-system path to the image.
    target_size:
        ``(width, height)`` for resizing.
    grayscale:
        If ``True``, load as single-channel grayscale.
    apply_clahe_enhancement:
        Whether to run CLAHE before resizing.

    Returns
    -------
    np.ndarray or None
        Preprocessed image, or ``None`` if the file could not be read.
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, flag)
    if image is None:
        return None
    if apply_clahe_enhancement:
        image = apply_clahe(image)
    image = resize_image(image, target_size)
    return normalize_image(image)
