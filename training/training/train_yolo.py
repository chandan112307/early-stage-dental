"""YOLO object-detection training for dental caries localisation.

Uses the `ultralytics <https://docs.ultralytics.com>`_ library with a
YOLOv8 model.  The script expects the dataset to follow the YOLO
directory convention::

    dataset_root/
        images/
            train/
            val/
        labels/
            train/
            val/
        data.yaml

Run as a standalone script::

    python -m training.training.train_yolo \\
        --data /path/to/data.yaml \\
        --epochs 100 \\
        --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from training.configs.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    METRICS_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    SEED,
    YOLO_IMG_SIZE,
)


def train(
    data_yaml: str | Path,
    model_name: str = "yolov8n.pt",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    img_size: int = YOLO_IMG_SIZE[0],
    learning_rate: float = LEARNING_RATE,
    output_dir: str | Path = OUTPUT_DIR,
    model_dir: str | Path = MODEL_DIR,
    metrics_dir: str | Path = METRICS_DIR,
) -> None:
    """Train a YOLOv8 model for dental caries detection.

    Parameters
    ----------
    data_yaml:
        Path to the YOLO-format ``data.yaml`` file.
    model_name:
        Pretrained YOLO checkpoint (e.g. ``yolov8n.pt``, ``yolov8s.pt``).
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    img_size:
        Image dimension (square) used during training.
    learning_rate:
        Initial learning rate.
    output_dir:
        Root directory for YOLO run artefacts.
    model_dir:
        Directory to copy the best model weights.
    metrics_dir:
        Directory to save training metrics JSON.
    """
    # Lazy import so the module can be loaded without ultralytics installed
    from ultralytics import YOLO  # type: ignore[import-untyped]

    for d in (output_dir, model_dir, metrics_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ------------------------------------------------------------------
    # Load pretrained model
    # ------------------------------------------------------------------
    model = YOLO(model_name)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate,
        seed=SEED,
        project=str(output_dir),
        name=f"yolo_run_{timestamp}",
        exist_ok=True,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Save best weights to model_dir
    # ------------------------------------------------------------------
    best_weights = Path(output_dir) / f"yolo_run_{timestamp}" / "weights" / "best.pt"
    dest = Path(model_dir) / f"yolo_best_{timestamp}.pt"
    if best_weights.exists():
        import shutil
        shutil.copy2(best_weights, dest)
        print(f"[INFO] Best YOLO weights saved to {dest}")

    # ------------------------------------------------------------------
    # Persist metrics
    # ------------------------------------------------------------------
    metrics: dict = {}
    if results is not None and hasattr(results, "results_dict"):
        metrics = {
            k: float(v) if hasattr(v, "item") else v
            for k, v in results.results_dict.items()
        }

    metrics_path = Path(metrics_dir) / f"yolo_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] YOLO metrics saved to {metrics_path}")


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for dental caries detection.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml for YOLO training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO model name or path.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=YOLO_IMG_SIZE[0])
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    parser.add_argument("--metrics-dir", type=str, default=str(METRICS_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        metrics_dir=args.metrics_dir,
    )
