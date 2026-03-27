"""Centralized dataset pipeline for dental caries detection training.

This module is the **single source of truth** for dataset acquisition,
conversion, and validation.  Every training module must call
:func:`ensure_dataset` before accessing any data.

The pipeline:
1. Downloads the dataset if not already present.
2. Converts raw formats (e.g. Supervisely) into the canonical structure.
3. Validates that the required directories and files exist.
4. Halts execution immediately if validation fails.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API — the ONLY dataset entry-point
# ------------------------------------------------------------------

def ensure_dataset(dataset_path: Path) -> Path:
    """Download, convert, and validate the dataset.

    This function is the **only** location where dataset download,
    conversion, and validation occur.  It must be called before any
    dataset access in every training module.

    Parameters
    ----------
    dataset_path:
        Target directory for the dataset.

    Returns
    -------
    Path
        The validated dataset root directory.

    Raises
    ------
    SystemExit
        If the dataset cannot be acquired or fails validation.
    """
    # 1. Download if not already present
    if not (dataset_path.exists() and any(dataset_path.iterdir())):
        _download_dataset(dataset_path)

    # 2. Convert if needed (e.g. Supervisely → canonical layout)
    _convert_if_needed(dataset_path)

    # 3. Validate — hard gate, no recovery
    _validate_dataset(dataset_path)

    print(f"[INFO] Dataset ready at {dataset_path}")
    return dataset_path


# ------------------------------------------------------------------
# Download
# ------------------------------------------------------------------

def _download_dataset(dataset_path: Path) -> None:
    """Download the DentalAI dataset via dataset-tools.  Terminates on failure.

    This is the **only** dataset acquisition logic in the entire codebase.
    There are no alternative sources, no fallbacks, and no retry with
    other providers.  If the download fails, execution stops immediately.
    """
    try:
        import dataset_tools as dtools  # type: ignore[import-untyped]

        print("[INFO] Downloading DentalAI dataset via dataset-tools …")
        dataset_path.mkdir(parents=True, exist_ok=True)
        dtools.download(dataset="Dentalai", dst_dir=str(dataset_path))
        print(f"[INFO] Dataset downloaded to {dataset_path}")
    except Exception as exc:
        print(f"[FATAL] Dataset acquisition failed: {exc}")
        print("[FATAL] Could not acquire dataset. Terminating.")
        sys.exit(1)


# ------------------------------------------------------------------
# Conversion
# ------------------------------------------------------------------

def _convert_if_needed(dataset_path: Path) -> None:
    """Convert raw dataset formats into the canonical structure.

    The canonical structure contains::

        classification/caries/  classification/no_caries/
        detection/images/  detection/labels/  detection/data.yaml
        segmentation/images/  segmentation/masks/
    """
    # Already converted
    if (dataset_path / "classification").exists():
        return

    # Check for Supervisely format (img + ann directories)
    img_dirs = list(dataset_path.rglob("img"))
    ann_dirs = list(dataset_path.rglob("ann"))

    if img_dirs and ann_dirs:
        _convert_supervisely(dataset_path, img_dirs[0], ann_dirs[0])


def _convert_supervisely(
    dataset_path: Path, images_dir: Path, ann_dir: Path
) -> None:
    """Convert Supervisely (DentalAI) format into canonical structure."""
    print("[INFO] Converting Supervisely dataset …")

    cls_dir = dataset_path / "classification"
    det_img_dir = dataset_path / "detection" / "images"
    det_lbl_dir = dataset_path / "detection" / "labels"
    seg_img_dir = dataset_path / "segmentation" / "images"
    seg_mask_dir = dataset_path / "segmentation" / "masks"

    for d in [cls_dir, det_img_dir, det_lbl_dir, seg_img_dir, seg_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    (cls_dir / "caries").mkdir(exist_ok=True)
    (cls_dir / "no_caries").mkdir(exist_ok=True)

    import cv2
    import numpy as np

    for ann_file in ann_dir.glob("*.json"):
        with open(ann_file) as f:
            ann = json.load(f)

        img_name = ann_file.stem + ".jpg"
        img_path = images_dir / img_name
        if not img_path.exists():
            continue

        has_caries = False

        shutil.copy(img_path, seg_img_dir / img_name)
        shutil.copy(img_path, det_img_dir / img_name)

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        yolo_labels: list[str] = []

        for obj in ann.get("objects", []):
            label = obj.get("classTitle", "").lower()
            if "caries" in label:
                has_caries = True
                points = obj.get("points", {}).get("exterior", [])
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = abs(x2 - x1) / w
                    bh = abs(y2 - y1) / h
                    yolo_labels.append(f"0 {xc} {yc} {bw} {bh}")
                    mask[int(y1):int(y2), int(x1):int(x2)] = 255

        with open(det_lbl_dir / (ann_file.stem + ".txt"), "w") as f:
            f.write("\n".join(yolo_labels))

        cv2.imwrite(str(seg_mask_dir / img_name), mask)

        if has_caries:
            shutil.copy(img_path, cls_dir / "caries" / img_name)
        else:
            shutil.copy(img_path, cls_dir / "no_caries" / img_name)

    yaml_content = (
        f"train: {det_img_dir}\n"
        f"val: {det_img_dir}\n"
        f"nc: 1\n"
        f"names: ['caries']\n"
    )
    with open(dataset_path / "detection" / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("[INFO] Conversion complete")


# ------------------------------------------------------------------
# Validation — hard gate
# ------------------------------------------------------------------

def _validate_dataset(dataset_path: Path) -> None:
    """Validate that the dataset has the required structure.

    Checks that at least one task directory (classification, detection,
    or segmentation) exists and contains files.  Terminates execution
    on failure — no warnings, no recovery.
    """
    if not dataset_path.exists():
        print(f"[FATAL] Dataset directory does not exist: {dataset_path}")
        sys.exit(1)

    # Check that the dataset is non-empty
    if not any(dataset_path.iterdir()):
        print(f"[FATAL] Dataset directory is empty: {dataset_path}")
        sys.exit(1)

    # At least one task must have content
    task_found = False

    # Classification check
    cls_dir = dataset_path / "classification"
    if cls_dir.is_dir() and any(cls_dir.iterdir()):
        task_found = True

    # Also accept flat class directories (No Caries / Caries)
    for class_name in ("Caries", "caries", "No Caries", "no_caries"):
        if (dataset_path / class_name).is_dir():
            task_found = True
            break

    # Detection check
    det_dir = dataset_path / "detection"
    if det_dir.is_dir():
        data_yaml = det_dir / "data.yaml"
        images_dir = det_dir / "images"
        if data_yaml.exists() and images_dir.is_dir():
            task_found = True

    # Segmentation check
    seg_dir = dataset_path / "segmentation"
    if seg_dir.is_dir():
        seg_imgs = seg_dir / "images"
        seg_masks = seg_dir / "masks"
        if seg_imgs.is_dir() and seg_masks.is_dir():
            task_found = True

    if not task_found:
        print(
            f"[FATAL] Dataset at {dataset_path} does not contain any "
            f"recognized task directories (classification/, detection/, "
            f"segmentation/) or class folders. Terminating."
        )
        sys.exit(1)
