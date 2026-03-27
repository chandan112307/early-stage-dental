"""Model evaluation and metrics computation for dental caries detection.

Computes standard classification metrics (accuracy, precision, recall, F1,
confusion matrix) and persists them to JSON for downstream dashboards or
CI comparison.

Run as a standalone script::

    python -m training.evaluation.evaluate \\
        --model-path models/mobilenet_best.keras \\
        --data-dir /path/to/dataset \\
        --model-type mobilenet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from training.configs.config import (
    CLASS_NAMES,
    METRICS_DIR,
    MOBILENET_IMG_SIZE,
    UNET_IMG_SIZE,
)


# ------------------------------------------------------------------
# Core metric helpers
# ------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, F1, and confusion matrix.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels.
    class_names:
        Human-readable class names for the report.

    Returns
    -------
    dict
        Dictionary with all computed metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    class_names = class_names or list(CLASS_NAMES)
    num_classes = len(class_names)
    average = "binary" if num_classes == 2 else "weighted"

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    recall = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "class_names": class_names,
        "num_samples": int(len(y_true)),
    }


def compute_segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute IoU (Jaccard) and Dice score for binary segmentation masks.

    Parameters
    ----------
    y_true:
        Ground-truth binary masks ``(N, H, W)`` or ``(N, H, W, 1)``.
    y_pred:
        Predicted probability maps (same shape).
    threshold:
        Binarisation threshold for predictions.

    Returns
    -------
    dict
        ``{"iou": …, "dice": …, "pixel_accuracy": …}``
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_bin.flatten()

    intersection = float(np.sum(y_true_flat * y_pred_flat))
    union = float(np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection)

    iou = intersection / union if union > 0 else 0.0
    dice = (2.0 * intersection) / (float(np.sum(y_true_flat)) + float(np.sum(y_pred_flat))) \
        if (np.sum(y_true_flat) + np.sum(y_pred_flat)) > 0 else 0.0
    pixel_acc = float(np.mean(y_true_flat == y_pred_flat))

    return {
        "iou": iou,
        "dice": dice,
        "pixel_accuracy": pixel_acc,
    }


# ------------------------------------------------------------------
# Save / load helpers
# ------------------------------------------------------------------

def save_metrics(
    metrics: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """Serialise metrics dictionary to a JSON file.

    Parameters
    ----------
    metrics:
        Metrics dictionary (must be JSON-serialisable).
    output_path:
        Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[INFO] Metrics saved to {output_path}")


# ------------------------------------------------------------------
# High-level evaluation drivers
# ------------------------------------------------------------------

def evaluate_mobilenet(
    model_path: str | Path,
    data_dir: str | Path,
    metrics_dir: str | Path = METRICS_DIR,
) -> Dict[str, Any]:
    """Evaluate a saved MobileNet model on the test split.

    Parameters
    ----------
    model_path:
        Path to the saved ``.keras`` model file.
    data_dir:
        Root of the dataset directory.
    metrics_dir:
        Where to save the evaluation JSON.

    Returns
    -------
    dict
        Computed metrics.
    """
    import tensorflow as tf

    from training.data.dataset import DentalDataset

    model = tf.keras.models.load_model(model_path)

    dataset = DentalDataset(root_dir=data_dir, target_size=MOBILENET_IMG_SIZE)
    _, _, (test_paths, test_labels) = dataset.split()
    X_test, y_test = dataset.load_images(test_paths, test_labels)

    print(f"[INFO] Evaluating on {len(X_test)} test images …")

    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob.squeeze() >= 0.5).astype(int)

    metrics = compute_classification_metrics(y_test, y_pred)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_metrics(
        metrics, Path(metrics_dir) / f"mobilenet_eval_{timestamp}.json"
    )

    _print_summary(metrics)
    return metrics


def evaluate_unet(
    model_path: str | Path,
    image_dir: str | Path,
    mask_dir: str | Path,
    metrics_dir: str | Path = METRICS_DIR,
) -> Dict[str, Any]:
    """Evaluate a saved U-Net model on validation data.

    Parameters
    ----------
    model_path:
        Path to the saved ``.keras`` model file.
    image_dir:
        Directory with test images.
    mask_dir:
        Directory with ground-truth masks.
    metrics_dir:
        Where to save the evaluation JSON.

    Returns
    -------
    dict
        Computed segmentation metrics.
    """
    import tensorflow as tf

    from training.training.train_unet import load_segmentation_data

    model = tf.keras.models.load_model(model_path)

    _, (X_val, y_val) = load_segmentation_data(image_dir, mask_dir)
    print(f"[INFO] Evaluating on {len(X_val)} validation images …")

    y_pred = model.predict(X_val, verbose=0)
    metrics = compute_segmentation_metrics(y_val, y_pred)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_metrics(
        metrics, Path(metrics_dir) / f"unet_eval_{timestamp}.json"
    )

    _print_summary(metrics)
    return metrics


def _print_summary(metrics: Dict[str, Any]) -> None:
    """Pretty-print a metrics summary to stdout."""
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    for key, value in metrics.items():
        if key in ("classification_report", "confusion_matrix", "class_names"):
            continue
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    print("=" * 50 + "\n")


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained dental caries model.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model file (.keras or .pt).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root dataset directory (for classification) or image directory (for segmentation).",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help="Mask directory (required for U-Net segmentation evaluation).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["mobilenet", "unet"],
        required=True,
        help="Type of model to evaluate.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=str(METRICS_DIR),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model_type == "mobilenet":
        evaluate_mobilenet(
            model_path=args.model_path,
            data_dir=args.data_dir,
            metrics_dir=args.metrics_dir,
        )
    elif args.model_type == "unet":
        if not args.mask_dir:
            print("[ERROR] --mask-dir is required for U-Net evaluation.")
            sys.exit(1)
        evaluate_unet(
            model_path=args.model_path,
            image_dir=args.data_dir,
            mask_dir=args.mask_dir,
            metrics_dir=args.metrics_dir,
        )
