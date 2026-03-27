"""Fully automated end-to-end training + inference pipeline.

Run the complete pipeline with **zero manual intervention**::

    python -m training                     # Train all models, export, deploy
    python -m training --model mobilenet   # MobileNet only
    python -m training --model yolo        # YOLO only
    python -m training --model unet        # U-Net only
    python -m training --skip-export       # Train only, skip ONNX export
    python -m training --skip-deploy       # Train + export, skip backend deploy

The pipeline automatically:
1. Downloads the dataset (if not present)
2. Trains the selected model(s)
3. Exports trained models to ONNX format
4. Deploys ONNX models to ``backend/models/``
5. Aggregates evaluation metrics for the backend
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from training.configs.config import (
    BACKEND_MODEL_DIR,
    BATCH_SIZE,
    DATASET_DIR,
    EPOCHS,
    KAGGLE_DATASET_NAME,
    LEARNING_RATE,
    METRICS_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
)
from training.data.dataset_utils import ensure_dataset


def _banner(msg: str) -> None:
    """Print a prominent stage banner."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width + "\n")


def run_pipeline(
    models: list[str] | None = None,
    dataset: str | None = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    skip_export: bool = False,
    skip_deploy: bool = False,
) -> None:
    """Execute the end-to-end training → export → deploy pipeline.

    Parameters
    ----------
    models:
        List of models to train (``"mobilenet"``, ``"yolo"``, ``"unet"``).
        Defaults to all three.
    dataset:
        Explicit dataset path.  Auto-downloaded when ``None``.
    epochs:
        Training epochs.
    batch_size:
        Mini-batch size.
    learning_rate:
        Initial learning rate.
    skip_export:
        If ``True``, skip the ONNX export stage.
    skip_deploy:
        If ``True``, skip deploying to ``backend/models/``.
    """
    if models is None:
        models = ["mobilenet", "yolo", "unet"]

    # ------------------------------------------------------------------
    # Stage 1: Dataset
    # ------------------------------------------------------------------
    _banner("STAGE 1 / 4 — Dataset")
    if dataset:
        ds_root = Path(dataset)
        print(f"[INFO] Using provided dataset at {ds_root}")
    else:
        ds_root = ensure_dataset(DATASET_DIR, KAGGLE_DATASET_NAME)

    # ------------------------------------------------------------------
    # Stage 2: Training
    # ------------------------------------------------------------------
    _banner("STAGE 2 / 4 — Training")
    trained_paths: dict[str, Path] = {}

    if "mobilenet" in models:
        trained_paths["mobilenet"] = _train_mobilenet(
            ds_root, epochs, batch_size, learning_rate,
        )

    if "yolo" in models:
        yolo_path = _train_yolo(ds_root, epochs, batch_size, learning_rate)
        if yolo_path is not None:
            trained_paths["yolo"] = yolo_path

    if "unet" in models:
        trained_paths["unet"] = _train_unet(
            ds_root, epochs, batch_size, learning_rate,
        )

    print(f"\n[INFO] Training complete.  Models saved: {list(trained_paths.keys())}")

    # ------------------------------------------------------------------
    # Stage 3: ONNX Export
    # ------------------------------------------------------------------
    if skip_export:
        print("\n[INFO] Skipping ONNX export (--skip-export)")
    else:
        _banner("STAGE 3 / 4 — ONNX Export")
        from training.export.export_onnx import export_all

        export_all(
            mobilenet_path=trained_paths.get("mobilenet"),
            yolo_path=trained_paths.get("yolo"),
            unet_path=trained_paths.get("unet"),
            output_dir=MODEL_DIR,
        )

    # ------------------------------------------------------------------
    # Stage 4: Deploy to backend
    # ------------------------------------------------------------------
    if skip_deploy or skip_export:
        if skip_deploy:
            print("\n[INFO] Skipping deployment (--skip-deploy)")
        return

    _banner("STAGE 4 / 4 — Deploy to Backend")
    from training.export.deploy import aggregate_metrics, deploy_models

    deploy_models(source_dir=MODEL_DIR, target_dir=BACKEND_MODEL_DIR)
    aggregate_metrics(metrics_dir=METRICS_DIR, target_dir=BACKEND_MODEL_DIR)

    print("\n[INFO] Pipeline complete.  Backend is ready for inference.")


# ------------------------------------------------------------------
# Per-model training wrappers
# ------------------------------------------------------------------

def _train_mobilenet(
    ds_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Path:
    """Train MobileNet classifier and return the checkpoint path."""
    print("\n--- Training MobileNet classifier ---")
    from training.training.train_mobilenet import train

    return train(
        data_dir=ds_root,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        metrics_dir=METRICS_DIR,
    )


def _train_yolo(
    ds_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Path | None:
    """Train YOLO detector and return the checkpoint path."""
    print("\n--- Training YOLO detector ---")
    from training.training.train_yolo import _find_data_yaml, train

    try:
        data_yaml = _find_data_yaml(ds_root)
    except FileNotFoundError:
        print("[WARN] No data.yaml found — skipping YOLO training")
        return None

    return train(
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        metrics_dir=METRICS_DIR,
    )


def _train_unet(
    ds_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Path:
    """Train U-Net segmentor and return the checkpoint path."""
    print("\n--- Training U-Net segmentor ---")
    from training.training.train_unet import _find_subdir, train

    image_dir = _find_subdir(ds_root, "images", "image", "imgs", "img")
    mask_dir = _find_subdir(ds_root, "masks", "mask", "labels", "label")

    return train(
        image_dir=image_dir,
        mask_dir=mask_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        metrics_dir=METRICS_DIR,
    )


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fully automated dental caries training pipeline.  "
            "Downloads data → trains models → exports to ONNX → "
            "deploys to backend."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        choices=["mobilenet", "yolo", "unet"],
        default=None,
        help=(
            "Model(s) to train.  "
            "Defaults to all three (mobilenet, yolo, unet)."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Explicit dataset directory (auto-downloaded if omitted).",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export after training.",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip deploying models to backend/models/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        models=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        skip_export=args.skip_export,
        skip_deploy=args.skip_deploy,
    )
