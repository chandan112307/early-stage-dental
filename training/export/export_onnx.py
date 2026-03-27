"""Export trained models to ONNX format for backend inference.

Supports:
- **MobileNet** (.keras → ONNX via tf2onnx)
- **YOLOv8** (.pt → ONNX via ultralytics export)
- **U-Net** (.keras → ONNX via tf2onnx)

Run standalone::

    python -m training.export.export_onnx \\
        --mobilenet models/mobilenet_best.keras \\
        --yolo models/yolo_best.pt \\
        --unet models/unet_best.keras
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from training.configs.config import (
    MODEL_DIR,
    MOBILENET_IMG_SIZE,
    UNET_IMG_SIZE,
    YOLO_IMG_SIZE,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logger = logging.getLogger(__name__)


def export_mobilenet_to_onnx(
    keras_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a MobileNet .keras model to ONNX.

    Parameters
    ----------
    keras_path:
        Path to the saved ``.keras`` model.
    output_path:
        Destination ``.onnx`` file.  Defaults to same directory with
        ``.onnx`` extension.

    Returns
    -------
    Path
        Path to the exported ONNX file.
    """
    import numpy as np
    import tensorflow as tf

    keras_path = Path(keras_path)
    if output_path is None:
        output_path = keras_path.with_suffix(".onnx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading MobileNet from {keras_path} …")
    model = tf.keras.models.load_model(keras_path)

    # Build a concrete function with a fixed input signature
    h, w = MOBILENET_IMG_SIZE
    input_spec = tf.TensorSpec((1, h, w, 3), tf.float32, name="input")

    @tf.function(input_signature=[input_spec])
    def _forward(x: tf.Tensor) -> tf.Tensor:
        return model(x, training=False)

    # Use tf2onnx for conversion
    import tf2onnx

    concrete = _forward.get_concrete_function()
    model_proto, _ = tf2onnx.convert.from_function(
        concrete,
        input_signature=[input_spec],
        output_path=str(output_path),
    )

    print(f"[INFO] MobileNet ONNX exported to {output_path}")
    return output_path


def export_yolo_to_onnx(
    pt_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a YOLOv8 .pt model to ONNX.

    Parameters
    ----------
    pt_path:
        Path to the saved ``.pt`` model weights.
    output_path:
        Destination ``.onnx`` file.  Defaults to same directory.

    Returns
    -------
    Path
        Path to the exported ONNX file.
    """
    from ultralytics import YOLO  # type: ignore[import-untyped]

    pt_path = Path(pt_path)

    print(f"[INFO] Loading YOLO from {pt_path} …")
    model = YOLO(str(pt_path))

    # ultralytics .export() returns the path to the exported file
    exported = model.export(format="onnx", imgsz=YOLO_IMG_SIZE[0])
    exported_path = Path(exported)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if exported_path != output_path:
            import shutil
            shutil.move(str(exported_path), str(output_path))
            exported_path = output_path

    print(f"[INFO] YOLO ONNX exported to {exported_path}")
    return exported_path


def export_unet_to_onnx(
    keras_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a U-Net .keras model to ONNX.

    Parameters
    ----------
    keras_path:
        Path to the saved ``.keras`` model.
    output_path:
        Destination ``.onnx`` file.

    Returns
    -------
    Path
        Path to the exported ONNX file.
    """
    import numpy as np
    import tensorflow as tf

    keras_path = Path(keras_path)
    if output_path is None:
        output_path = keras_path.with_suffix(".onnx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading U-Net from {keras_path} …")
    model = tf.keras.models.load_model(keras_path)

    h, w = UNET_IMG_SIZE
    input_spec = tf.TensorSpec((1, h, w, 3), tf.float32, name="input")

    @tf.function(input_signature=[input_spec])
    def _forward(x: tf.Tensor) -> tf.Tensor:
        return model(x, training=False)

    import tf2onnx

    concrete = _forward.get_concrete_function()
    model_proto, _ = tf2onnx.convert.from_function(
        concrete,
        input_signature=[input_spec],
        output_path=str(output_path),
    )

    print(f"[INFO] U-Net ONNX exported to {output_path}")
    return output_path


def export_all(
    mobilenet_path: Optional[str | Path] = None,
    yolo_path: Optional[str | Path] = None,
    unet_path: Optional[str | Path] = None,
    output_dir: str | Path = MODEL_DIR,
) -> dict[str, Path]:
    """Export all available trained models to ONNX.

    Parameters
    ----------
    mobilenet_path:
        Path to MobileNet ``.keras`` file (skipped if ``None``).
    yolo_path:
        Path to YOLO ``.pt`` file (skipped if ``None``).
    unet_path:
        Path to U-Net ``.keras`` file (skipped if ``None``).
    output_dir:
        Directory for ONNX output files.

    Returns
    -------
    dict
        Mapping of model name → ONNX path for successfully exported models.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}

    if mobilenet_path is not None:
        try:
            onnx_path = export_mobilenet_to_onnx(
                mobilenet_path,
                output_dir / "mobilenet_classifier.onnx",
            )
            exported["mobilenet"] = onnx_path
        except Exception as exc:
            print(f"[ERROR] MobileNet ONNX export failed: {exc}")

    if yolo_path is not None:
        try:
            onnx_path = export_yolo_to_onnx(
                yolo_path,
                output_dir / "yolo_detector.onnx",
            )
            exported["yolo"] = onnx_path
        except Exception as exc:
            print(f"[ERROR] YOLO ONNX export failed: {exc}")

    if unet_path is not None:
        try:
            onnx_path = export_unet_to_onnx(
                unet_path,
                output_dir / "unet_segmentor.onnx",
            )
            exported["unet"] = onnx_path
        except Exception as exc:
            print(f"[ERROR] U-Net ONNX export failed: {exc}")

    return exported


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export trained dental models to ONNX format.",
    )
    parser.add_argument(
        "--mobilenet", type=str, default=None,
        help="Path to MobileNet .keras model.",
    )
    parser.add_argument(
        "--yolo", type=str, default=None,
        help="Path to YOLO .pt model.",
    )
    parser.add_argument(
        "--unet", type=str, default=None,
        help="Path to U-Net .keras model.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(MODEL_DIR),
        help="Directory for ONNX output files.",
    )
    args = parser.parse_args()

    results = export_all(
        mobilenet_path=args.mobilenet,
        yolo_path=args.yolo,
        unet_path=args.unet,
        output_dir=args.output_dir,
    )

    if results:
        print(f"\n[INFO] Exported {len(results)} model(s) to ONNX:")
        for name, path in results.items():
            print(f"  - {name}: {path}")
    else:
        print("[WARN] No models were exported.")
