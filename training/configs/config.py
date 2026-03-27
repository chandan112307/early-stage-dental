"""Configuration constants for the dental caries detection training pipeline.

This module centralises every tuneable parameter and path used across
the data-loading, preprocessing, training, and evaluation stages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # training/

DATA_DIR: Path = _PROJECT_ROOT / "data"
OUTPUT_DIR: Path = _PROJECT_ROOT / "outputs"
MODEL_DIR: Path = _PROJECT_ROOT / "models"
METRICS_DIR: Path = _PROJECT_ROOT / "metrics"

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = ["No Caries", "Caries"]
NUM_CLASSES: int = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Image dimensions per model architecture
# ---------------------------------------------------------------------------
MOBILENET_IMG_SIZE: Tuple[int, int] = (224, 224)
YOLO_IMG_SIZE: Tuple[int, int] = (640, 640)
UNET_IMG_SIZE: Tuple[int, int] = (256, 256)

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 16
EPOCHS: int = 50
LEARNING_RATE: float = 1e-4
EARLY_STOPPING_PATIENCE: int = 7
REDUCE_LR_PATIENCE: int = 3
REDUCE_LR_FACTOR: float = 0.5
MIN_LR: float = 1e-7

# ---------------------------------------------------------------------------
# Data split ratios  (must sum to 1.0)
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ---------------------------------------------------------------------------
# Preprocessing – CLAHE parameters
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)

# ---------------------------------------------------------------------------
# Augmentation limits
# ---------------------------------------------------------------------------
ROTATION_RANGE: int = 15
BRIGHTNESS_RANGE: Tuple[float, float] = (0.8, 1.2)
CONTRAST_RANGE: Tuple[float, float] = (0.8, 1.2)
HORIZONTAL_FLIP: bool = True
VERTICAL_FLIP: bool = False  # dental X-rays are orientation-sensitive

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42


@dataclass
class TrainingConfig:
    """Aggregate configuration object for a single training run.

    This is a convenience wrapper so that downstream code can receive a
    single config object instead of importing individual constants.
    """

    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))
    num_classes: int = NUM_CLASSES

    # Paths
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    model_dir: Path = MODEL_DIR
    metrics_dir: Path = METRICS_DIR

    # Image sizes
    mobilenet_img_size: Tuple[int, int] = MOBILENET_IMG_SIZE
    yolo_img_size: Tuple[int, int] = YOLO_IMG_SIZE
    unet_img_size: Tuple[int, int] = UNET_IMG_SIZE

    # Training
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    reduce_lr_patience: int = REDUCE_LR_PATIENCE
    reduce_lr_factor: float = REDUCE_LR_FACTOR
    min_lr: float = MIN_LR

    # Splits
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO

    # Seed
    seed: int = SEED

    def ensure_dirs(self) -> None:
        """Create output directories if they do not exist."""
        for d in (self.output_dir, self.model_dir, self.metrics_dir):
            d.mkdir(parents=True, exist_ok=True)
