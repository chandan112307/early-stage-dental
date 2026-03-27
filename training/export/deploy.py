"""Deploy ONNX models and metrics to the backend directory.

Copies exported ONNX files from ``training/models/`` to
``backend/models/`` with the canonical names expected by the backend,
and aggregates training metrics into a single
``evaluation_metrics.json`` file.

Run standalone::

    python -m training.export.deploy
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from training.configs.config import (
    BACKEND_METRICS_FILE,
    BACKEND_MODEL_DIR,
    METRICS_DIR,
    MODEL_DIR,
    ONNX_CLASSIFIER_NAME,
    ONNX_DETECTOR_NAME,
    ONNX_SEGMENTOR_NAME,
)


def deploy_models(
    source_dir: str | Path = MODEL_DIR,
    target_dir: str | Path = BACKEND_MODEL_DIR,
) -> list[str]:
    """Copy ONNX models from *source_dir* to *target_dir*.

    Parameters
    ----------
    source_dir:
        Directory containing exported ``.onnx`` files.
    target_dir:
        Backend model directory.

    Returns
    -------
    list[str]
        Names of successfully deployed models.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    deployed: list[str] = []

    model_map = {
        ONNX_CLASSIFIER_NAME: "MobileNet classifier",
        ONNX_DETECTOR_NAME: "YOLO detector",
        ONNX_SEGMENTOR_NAME: "U-Net segmentor",
    }

    for filename, label in model_map.items():
        src = source_dir / filename
        dst = target_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[INFO] Deployed {label}: {dst}")
            deployed.append(filename)
        else:
            print(f"[WARN] {label} not found at {src} – skipping")

    return deployed


def aggregate_metrics(
    metrics_dir: str | Path = METRICS_DIR,
    target_dir: str | Path = BACKEND_MODEL_DIR,
) -> Path:
    """Aggregate individual training metric files into one JSON for the backend.

    Scans ``metrics_dir`` for the most recent metric files for each model
    and combines them into ``evaluation_metrics.json``.

    Parameters
    ----------
    metrics_dir:
        Source directory with per-model JSON metric files.
    target_dir:
        Where to write the aggregated ``evaluation_metrics.json``.

    Returns
    -------
    Path
        Path to the written metrics file.
    """
    metrics_dir = Path(metrics_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    aggregated: dict[str, dict] = {}

    # Find the most recent file matching each prefix
    prefixes = {
        "classifier": ["mobilenet_eval_", "mobilenet_history_"],
        "detector": ["yolo_metrics_"],
        "segmentor": ["unet_eval_", "unet_history_"],
    }

    for model_key, prefix_list in prefixes.items():
        best_file: Path | None = None
        for prefix in prefix_list:
            candidates = sorted(
                metrics_dir.glob(f"{prefix}*.json"), reverse=True,
            )
            # Prefer evaluation files over history files
            if candidates:
                if best_file is None or "eval" in prefix:
                    best_file = candidates[0]

        if best_file is not None:
            try:
                data = json.loads(best_file.read_text())
                aggregated[model_key] = _normalise_metrics(model_key, data)
                print(f"[INFO] Aggregated {model_key} metrics from {best_file.name}")
            except Exception as exc:
                print(f"[WARN] Failed to read {best_file}: {exc}")

    output_path = target_dir / BACKEND_METRICS_FILE
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"[INFO] Evaluation metrics written to {output_path}")
    return output_path


def _normalise_metrics(model_key: str, data: dict) -> dict:
    """Extract the relevant metric fields for the backend."""
    if model_key == "classifier":
        # From evaluate.py → classification report (scalar values)
        if "accuracy" in data and not isinstance(data["accuracy"], list):
            return {
                "accuracy": data.get("accuracy"),
                "precision": data.get("precision"),
                "recall": data.get("recall"),
                "f1_score": data.get("f1_score"),
            }
        # Fallback: training history (last epoch values)
        return {
            "accuracy": _last(data.get("val_accuracy", data.get("accuracy"))),
        }

    if model_key == "detector":
        # From YOLO metrics
        return {
            "mAP50": data.get("metrics/mAP50(B)", data.get("mAP50")),
            "mAP50_95": data.get("metrics/mAP50-95(B)", data.get("mAP50_95")),
            "precision": data.get("metrics/precision(B)", data.get("precision")),
            "recall": data.get("metrics/recall(B)", data.get("recall")),
        }

    if model_key == "segmentor":
        if "dice" in data or "iou" in data:
            return {
                "dice_coefficient": data.get("dice"),
                "iou": data.get("iou"),
                "pixel_accuracy": data.get("pixel_accuracy"),
            }
        return {
            "pixel_accuracy": _last(
                data.get("val_accuracy", data.get("accuracy"))
            ),
        }

    return data


def _last(seq: list | float | None) -> float | None:
    """Return the last element if *seq* is a non-empty list, else return as-is."""
    if isinstance(seq, list):
        return seq[-1] if seq else None
    return seq


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deploy ONNX models and metrics to the backend.",
    )
    parser.add_argument(
        "--source-dir", type=str, default=str(MODEL_DIR),
        help="Directory containing ONNX model files.",
    )
    parser.add_argument(
        "--target-dir", type=str, default=str(BACKEND_MODEL_DIR),
        help="Backend models directory.",
    )
    parser.add_argument(
        "--metrics-dir", type=str, default=str(METRICS_DIR),
        help="Directory with per-model metrics JSON files.",
    )
    args = parser.parse_args()

    deploy_models(args.source_dir, args.target_dir)
    aggregate_metrics(args.metrics_dir, args.target_dir)
