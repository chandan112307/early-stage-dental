"""U-Net segmentation training for dental caries region segmentation.

Builds a lightweight U-Net encoder–decoder network with skip connections
using TensorFlow / Keras.  The model predicts a binary mask highlighting
carious regions in dental X-ray images.

Run as a standalone script::

    python -m training.training.train_unet \\
        --image-dir /path/to/images \\
        --mask-dir  /path/to/masks \\
        --epochs 50 \\
        --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from training.configs.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    METRICS_DIR,
    MIN_LR,
    MODEL_DIR,
    OUTPUT_DIR,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SEED,
    UNET_IMG_SIZE,
)

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------
# Model architecture
# ------------------------------------------------------------------

def _conv_block(
    inputs: tf.Tensor,
    num_filters: int,
) -> tf.Tensor:
    """Two consecutive Conv2D → BatchNorm → ReLU blocks."""
    x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def _encoder_block(
    inputs: tf.Tensor,
    num_filters: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Encoder block: conv_block → MaxPool.  Returns skip and pooled."""
    skip = _conv_block(inputs, num_filters)
    pool = layers.MaxPooling2D(2)(skip)
    return skip, pool


def _decoder_block(
    inputs: tf.Tensor,
    skip: tf.Tensor,
    num_filters: int,
) -> tf.Tensor:
    """Decoder block: UpConv → concatenate skip → conv_block."""
    x = layers.Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip])
    x = _conv_block(x, num_filters)
    return x


def build_unet_model(
    input_shape: Tuple[int, int, int] = (*UNET_IMG_SIZE, 3),
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """Build a U-Net model for binary segmentation.

    Parameters
    ----------
    input_shape:
        ``(H, W, C)`` of input images.
    learning_rate:
        Initial optimiser learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    s1, p1 = _encoder_block(inputs, 64)
    s2, p2 = _encoder_block(p1, 128)
    s3, p3 = _encoder_block(p2, 256)
    s4, p4 = _encoder_block(p3, 512)

    # Bottleneck
    b = _conv_block(p4, 1024)

    # Decoder
    d4 = _decoder_block(b, s4, 512)
    d3 = _decoder_block(d4, s3, 256)
    d2 = _decoder_block(d3, s2, 128)
    d1 = _decoder_block(d2, s1, 64)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)

    model = models.Model(inputs, outputs, name="UNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2)],
    )
    return model


# ------------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------------

def _discover_pairs(
    image_dir: Path,
    mask_dir: Path,
) -> Tuple[List[str], List[str]]:
    """Match images to their corresponding mask files by filename stem."""
    image_paths: List[str] = []
    mask_paths: List[str] = []

    mask_stems = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in _SUPPORTED_EXTENSIONS}

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        if img_path.stem in mask_stems:
            image_paths.append(str(img_path))
            mask_paths.append(str(mask_stems[img_path.stem]))

    return image_paths, mask_paths


def _load_pair(
    img_path: str,
    mask_path: str,
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess a single image–mask pair."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    mask = mask[..., np.newaxis]  # (H, W, 1)

    return img, mask


def load_segmentation_data(
    image_dir: str | Path,
    mask_dir: str | Path,
    target_size: Tuple[int, int] = UNET_IMG_SIZE,
    val_split: float = 0.15,
    seed: int = SEED,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Load images and masks, return train/val splits.

    Returns
    -------
    tuple
        ``((X_train, y_train), (X_val, y_val))``
    """
    image_paths, mask_paths = _discover_pairs(Path(image_dir), Path(mask_dir))
    if not image_paths:
        raise FileNotFoundError(
            f"No matching image-mask pairs found in {image_dir} / {mask_dir}"
        )

    images, masks = [], []
    for ip, mp in zip(image_paths, mask_paths):
        img, msk = _load_pair(ip, mp, target_size)
        images.append(img)
        masks.append(msk)

    X = np.array(images, dtype=np.float32)
    y = np.array(masks, dtype=np.float32)

    # Deterministic shuffle + split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    split_idx = int(len(X) * (1 - val_split))

    X_train, X_val = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_val = y[indices[:split_idx]], y[indices[split_idx:]]

    return (X_train, y_train), (X_val, y_val)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(
    image_dir: str | Path,
    mask_dir: str | Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    output_dir: str | Path = OUTPUT_DIR,
    model_dir: str | Path = MODEL_DIR,
    metrics_dir: str | Path = METRICS_DIR,
) -> None:
    """Run the full U-Net segmentation training loop.

    Parameters
    ----------
    image_dir:
        Directory containing training images.
    mask_dir:
        Directory containing corresponding binary masks.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    learning_rate:
        Initial learning rate.
    output_dir:
        Directory for TensorBoard logs.
    model_dir:
        Directory for model checkpoints.
    metrics_dir:
        Directory for training metrics JSON.
    """
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    for d in (output_dir, model_dir, metrics_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    (X_train, y_train), (X_val, y_val) = load_segmentation_data(
        image_dir, mask_dir
    )
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_unet_model(learning_rate=learning_rate)
    model.summary()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = str(Path(model_dir) / f"unet_best_{timestamp}.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(
            log_dir=str(Path(output_dir) / f"unet_logs_{timestamp}"),
        ),
    ]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics_path = Path(metrics_dir) / f"unet_history_{timestamp}.json"
    serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[INFO] Training history saved to {metrics_path}")
    print(f"[INFO] Best model saved to {checkpoint_path}")


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train U-Net for dental caries segmentation.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing training images.",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        required=True,
        help="Directory containing binary segmentation masks.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    parser.add_argument("--metrics-dir", type=str, default=str(METRICS_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        metrics_dir=args.metrics_dir,
    )
