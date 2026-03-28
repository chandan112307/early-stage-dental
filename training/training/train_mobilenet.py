"""MobileNet classification training for dental caries detection.

Uses TensorFlow / Keras with transfer learning from ImageNet weights.
The top classification head is replaced with a binary (or multi-class)
dense layer to predict ``No Caries`` vs ``Caries``.

Run as a standalone script::

    python -m training.training.train_mobilenet

All dataset preparation is handled automatically by the centralized
dataset pipeline — no dataset arguments are needed or accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF info logs

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

# Allow running as ``python -m training.training.train_mobilenet``
_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from training.configs.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATASET_DIR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    METRICS_DIR,
    MIN_LR,
    MOBILENET_IMG_SIZE,
    MODEL_DIR,
    NUM_CLASSES,
    OUTPUT_DIR,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SEED,
)
from training.data.dataset import DentalDataset
from training.data.dataset_utils import ensure_dataset


def build_mobilenet_model(
    input_shape: Tuple[int, int, int] = (*MOBILENET_IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    fine_tune_at: Optional[int] = 100,
) -> tf.keras.Model:
    """Build a MobileNetV2 model with a custom classification head.

    Parameters
    ----------
    input_shape:
        Shape of the input images ``(H, W, C)``.
    num_classes:
        Number of output classes.
    learning_rate:
        Initial optimiser learning rate.
    fine_tune_at:
        Layer index from which to unfreeze the base model for fine-tuning.
        If ``None``, the entire base is frozen.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # Freeze base layers up to ``fine_tune_at``
    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    else:
        base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(
                1 if num_classes == 2 else num_classes,
                activation="sigmoid" if num_classes == 2 else "softmax",
            ),
        ]
    )

    loss = (
        "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )

    return model


def train(
    data_dir: str | Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    output_dir: str | Path = OUTPUT_DIR,
    model_dir: str | Path = MODEL_DIR,
    metrics_dir: str | Path = METRICS_DIR,
) -> Path:
    """Run the full MobileNet training loop.

    Parameters
    ----------
    data_dir:
        Root of the dataset directory.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    learning_rate:
        Initial learning rate.
    output_dir:
        Directory for TensorBoard logs.
    model_dir:
        Directory to save the best model checkpoint.
    metrics_dir:
        Directory to save training metrics JSON.

    Returns
    -------
    Path
        Path to the saved best model checkpoint.
    """
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Ensure output dirs exist
    for d in (output_dir, model_dir, metrics_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = DentalDataset(
        root_dir=data_dir,
        class_names=["no_caries", "caries"],
        target_size=MOBILENET_IMG_SIZE,
    )
    print(f"[INFO] Discovered {dataset.num_samples} images: {dataset.class_distribution}")

    (train_paths, train_labels), (val_paths, val_labels), _ = dataset.split()

    X_train, y_train = dataset.load_images(train_paths, train_labels)
    X_val, y_val = dataset.load_images(val_paths, val_labels)

    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_mobilenet_model(learning_rate=learning_rate)
    # model.summary()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = str(Path(model_dir) / f"mobilenet_best_{timestamp}.keras")

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
            log_dir=str(Path(output_dir) / f"mobilenet_logs_{timestamp}"),
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
    metrics_path = Path(metrics_dir) / f"mobilenet_history_{timestamp}.json"
    serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[INFO] Training history saved to {metrics_path}")
    print(f"[INFO] Best model saved to {checkpoint_path}")

    return Path(checkpoint_path)


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 for dental caries classification.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Centralized dataset pipeline — always runs
    data_dir = ensure_dataset(DATASET_DIR)

    train(
        data_dir=data_dir / "classification",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
