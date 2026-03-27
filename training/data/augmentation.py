"""Data augmentation utilities for dental X-ray images.

These augmentations are designed to be **conservative** — dental X-rays are
orientation-sensitive and contain subtle textural cues, so aggressive
geometric or colour distortions can hurt model performance.

The module exposes both *individual* augmentation functions and a
ready-to-use :func:`build_augmentation_pipeline` that returns a composed
transformation compatible with NumPy arrays.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from training.configs.config import (
    BRIGHTNESS_RANGE,
    CONTRAST_RANGE,
    HORIZONTAL_FLIP,
    ROTATION_RANGE,
    SEED,
    VERTICAL_FLIP,
)


def random_rotation(
    image: np.ndarray,
    max_angle: int = ROTATION_RANGE,
) -> np.ndarray:
    """Rotate the image by a random angle in ``[-max_angle, max_angle]``.

    Parameters
    ----------
    image:
        Input image (H×W or H×W×C).
    max_angle:
        Maximum absolute rotation in degrees.

    Returns
    -------
    np.ndarray
        Rotated image with the same dimensions (border pixels filled black).
    """
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip the image horizontally with probability *p*."""
    if random.random() < p:
        return cv2.flip(image, 1)
    return image


def random_vertical_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip the image vertically with probability *p*."""
    if random.random() < p:
        return cv2.flip(image, 0)
    return image


def random_brightness(
    image: np.ndarray,
    brightness_range: Tuple[float, float] = BRIGHTNESS_RANGE,
) -> np.ndarray:
    """Randomly adjust image brightness by a multiplicative factor.

    Parameters
    ----------
    image:
        Input image (``float32`` in ``[0, 1]`` or ``uint8``).
    brightness_range:
        ``(low, high)`` multiplicative factor range.

    Returns
    -------
    np.ndarray
        Brightness-adjusted image, clipped to valid range.
    """
    factor = random.uniform(*brightness_range)
    adjusted = image.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255 if image.dtype == np.uint8 else 1.0).astype(
        image.dtype
    )


def random_contrast(
    image: np.ndarray,
    contrast_range: Tuple[float, float] = CONTRAST_RANGE,
) -> np.ndarray:
    """Randomly adjust image contrast.

    Contrast is adjusted by linearly interpolating between the mean
    intensity and the original image.

    Parameters
    ----------
    image:
        Input image.
    contrast_range:
        ``(low, high)`` contrast factor range.

    Returns
    -------
    np.ndarray
        Contrast-adjusted image.
    """
    factor = random.uniform(*contrast_range)
    mean = np.mean(image)
    adjusted = mean + factor * (image.astype(np.float32) - mean)
    return np.clip(adjusted, 0, 255 if image.dtype == np.uint8 else 1.0).astype(
        image.dtype
    )


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 0.02,
) -> np.ndarray:
    """Add Gaussian noise to simulate sensor noise in X-ray captures.

    Parameters
    ----------
    image:
        Input image (``float32`` expected, values in ``[0, 1]``).
    mean:
        Mean of the Gaussian distribution.
    std:
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    np.ndarray
        Noisy image clipped to ``[0, 1]``.
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0.0, 1.0)


# ------------------------------------------------------------------
# Pipeline builder
# ------------------------------------------------------------------

def build_augmentation_pipeline(
    rotation: bool = True,
    h_flip: bool = HORIZONTAL_FLIP,
    v_flip: bool = VERTICAL_FLIP,
    brightness: bool = True,
    contrast: bool = True,
    noise: bool = True,
    seed: Optional[int] = SEED,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a composed augmentation function.

    Parameters
    ----------
    rotation:
        Enable random rotation.
    h_flip:
        Enable horizontal flip.
    v_flip:
        Enable vertical flip.
    brightness:
        Enable brightness jitter.
    contrast:
        Enable contrast jitter.
    noise:
        Enable Gaussian noise.
    seed:
        Random seed for reproducibility (``None`` = non-deterministic).

    Returns
    -------
    Callable
        A function ``augment(image) -> image`` that applies the selected
        augmentations **in a random order**.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    transforms: List[Callable[[np.ndarray], np.ndarray]] = []
    if rotation:
        transforms.append(random_rotation)
    if h_flip:
        transforms.append(random_horizontal_flip)
    if v_flip:
        transforms.append(random_vertical_flip)
    if brightness:
        transforms.append(random_brightness)
    if contrast:
        transforms.append(random_contrast)
    if noise:
        transforms.append(add_gaussian_noise)

    def _augment(image: np.ndarray) -> np.ndarray:
        shuffled = list(transforms)
        random.shuffle(shuffled)
        for fn in shuffled:
            image = fn(image)
        return image

    return _augment
