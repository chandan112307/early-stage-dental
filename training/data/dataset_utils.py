"""Centralized dataset pipeline for dental caries detection training.

This module is the **single source of truth** for dataset acquisition,
conversion, and validation. Every training module must call
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

_IMAGE_DIR_NAMES = ("img", "imgs", "image", "images")
_ANN_DIR_NAMES = ("ann", "anns", "annotation", "annotations")
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_LABEL_EXTENSIONS = {".txt"}
_TASK_ROOT_SEARCH_DEPTH = 4


# ------------------------------------------------------------------
# Public API — the ONLY dataset entry-point
# ------------------------------------------------------------------

def ensure_dataset(dataset_path: Path) -> Path:
    """Download, convert, and validate the dataset.

    This function is the **only** location where dataset download,
    conversion, and validation occur. It must be called before any
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

    # 2. Convert if needed (e.g. Supervisely -> canonical layout)
    task_root = _find_existing_task_root(dataset_path)
    if not _all_training_tasks_available(task_root):
        _convert_if_needed(dataset_path)
        task_root = _find_existing_task_root(dataset_path)

    # 3. Validate — hard gate, no recovery
    task_root = task_root or dataset_path
    _validate_dataset(task_root)

    print(f"[INFO] Dataset ready at {task_root}")
    return task_root


# ------------------------------------------------------------------
# Download
# ------------------------------------------------------------------

def _download_dataset(dataset_path: Path) -> None:
    """Download the DentalAI dataset via dataset-tools. Terminates on failure.

    This is the **only** dataset acquisition logic in the entire codebase.
    There are no alternative sources, no fallbacks, and no retry with
    other providers. If the download fails, execution stops immediately.
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
    if _all_training_tasks_available(_find_existing_task_root(dataset_path)):
        return

    pairs = _find_supervisely_pairs(dataset_path)
    if pairs:
        _convert_supervisely(dataset_path, pairs)


def _convert_supervisely(
    dataset_path: Path,
    pairs: list[tuple[Path, Path, str | None]],
) -> None:
    """Convert raw Supervisely folders into the canonical multi-task layout."""
    print(
        f"[INFO] Converting Supervisely dataset from "
        f"{len(pairs)} raw folder pair(s) …"
    )

    cls_dir = dataset_path / "classification"
    det_dir = dataset_path / "detection"
    det_img_train = det_dir / "images" / "train"
    det_img_val = det_dir / "images" / "val"
    det_img_test = det_dir / "images" / "test"
    det_lbl_train = det_dir / "labels" / "train"
    det_lbl_val = det_dir / "labels" / "val"
    det_lbl_test = det_dir / "labels" / "test"
    seg_img_dir = dataset_path / "segmentation" / "images"
    seg_mask_dir = dataset_path / "segmentation" / "masks"

    for directory in [
        cls_dir / "caries",
        cls_dir / "no_caries",
        det_img_train,
        det_img_val,
        det_img_test,
        det_lbl_train,
        det_lbl_val,
        det_lbl_test,
        seg_img_dir,
        seg_mask_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    items = _collect_supervisely_items(pairs)
    if not items:
        print("[WARN] Found raw Supervisely folders but no matching image/annotation pairs.")
        return

    assignments = _assign_detection_splits(items)
    used_names: set[str] = set()

    for (img_path, ann_file, _raw_split), det_split in zip(items, assignments):
        with open(ann_file) as f:
            ann = json.load(f)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_name = _unique_output_name(img_path, used_names)
        has_caries = False

        shutil.copy2(img_path, seg_img_dir / img_name)

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        yolo_labels: list[str] = []

        for obj in ann.get("objects", []):
            label = obj.get("classTitle", "").lower()
            if "caries" not in label:
                continue

            has_caries = True
            points = obj.get("points", {}).get("exterior", [])
            if len(points) < 2:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            xc = ((x_min + x_max) / 2) / w
            yc = ((y_min + y_max) / 2) / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            yolo_labels.append(f"0 {xc} {yc} {bw} {bh}")

            poly = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)

        det_img_dst = {
            "train": det_img_train,
            "val": det_img_val,
            "test": det_img_test,
        }[det_split]
        det_lbl_dst = {
            "train": det_lbl_train,
            "val": det_lbl_val,
            "test": det_lbl_test,
        }[det_split]

        shutil.copy2(img_path, det_img_dst / img_name)
        with open(det_lbl_dst / f"{Path(img_name).stem}.txt", "w") as f:
            f.write("\n".join(yolo_labels))

        cv2.imwrite(str(seg_mask_dir / img_name), mask)

        if has_caries:
            shutil.copy2(img_path, cls_dir / "caries" / img_name)
        else:
            shutil.copy2(img_path, cls_dir / "no_caries" / img_name)

    yaml_lines = [
        "train: images/train",
        "val: images/val",
    ]
    if _directory_has_files(det_img_test, _IMAGE_EXTENSIONS):
        yaml_lines.append("test: images/test")
    yaml_lines.extend(
        [
            "nc: 1",
            "names: ['caries']",
        ]
    )
    with open(det_dir / "data.yaml", "w") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print("[INFO] Conversion complete")


# ------------------------------------------------------------------
# Validation — hard gate
# ------------------------------------------------------------------

def _validate_dataset(dataset_path: Path) -> None:
    """Validate that the dataset has the required structure.

    Checks that at least one task directory (classification, detection,
    or segmentation) exists and contains files. Terminates execution
    on failure — no warnings, no recovery.
    """
    if not dataset_path.exists():
        print(f"[FATAL] Dataset directory does not exist: {dataset_path}")
        sys.exit(1)

    if not any(dataset_path.iterdir()):
        print(f"[FATAL] Dataset directory is empty: {dataset_path}")
        sys.exit(1)

    if not _has_task_content(dataset_path):
        print(
            f"[FATAL] Dataset at {dataset_path} does not contain any "
            f"recognized task directories (classification/, detection/, "
            f"segmentation/) or class folders. Terminating."
        )
        sys.exit(1)


def _all_training_tasks_available(dataset_path: Path | None) -> bool:
    """Return True when all model-specific task directories are present."""
    if dataset_path is None:
        return False
    return (
        _classification_ready(dataset_path)
        and _detection_ready(dataset_path)
        and _segmentation_ready(dataset_path)
    )


def _has_task_content(dataset_path: Path) -> bool:
    """Return True when at least one recognized task has usable files."""
    return (
        _classification_ready(dataset_path)
        or _detection_ready(dataset_path)
        or _segmentation_ready(dataset_path)
    )


def _classification_ready(dataset_path: Path) -> bool:
    """Check whether a classification task is available at this root."""
    for base_dir in (dataset_path / "classification", dataset_path):
        caries_dir = _first_existing_dir(base_dir, "caries", "Caries")
        no_caries_dir = _first_existing_dir(base_dir, "no_caries", "No Caries")
        if caries_dir and no_caries_dir:
            return (
                _directory_has_files(caries_dir, _IMAGE_EXTENSIONS)
                and _directory_has_files(no_caries_dir, _IMAGE_EXTENSIONS)
            )
    return False


def _detection_ready(dataset_path: Path) -> bool:
    """Check whether a detection task is available at this root."""
    det_dir = dataset_path / "detection"
    if not det_dir.is_dir():
        return False

    images_dir = det_dir / "images"
    labels_dir = det_dir / "labels"
    data_yaml = det_dir / "data.yaml"
    if not (data_yaml.exists() and images_dir.is_dir() and labels_dir.is_dir()):
        return False

    return (
        _directory_has_files(images_dir, _IMAGE_EXTENSIONS)
        and _directory_has_files(labels_dir, _LABEL_EXTENSIONS)
    )


def _segmentation_ready(dataset_path: Path) -> bool:
    """Check whether a segmentation task is available at this root."""
    seg_dir = dataset_path / "segmentation"
    if not seg_dir.is_dir():
        return False

    seg_imgs = seg_dir / "images"
    seg_masks = seg_dir / "masks"
    return (
        seg_imgs.is_dir()
        and seg_masks.is_dir()
        and _directory_has_files(seg_imgs, _IMAGE_EXTENSIONS)
        and _directory_has_files(seg_masks, _IMAGE_EXTENSIONS)
    )


def _directory_has_files(path: Path, suffixes: set[str]) -> bool:
    """Return True if *path* contains at least one matching file recursively."""
    if not path.is_dir():
        return False

    for candidate in path.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in suffixes:
            return True
    return False


def _first_existing_dir(base_dir: Path, *names: str) -> Path | None:
    """Return the first matching directory under *base_dir*."""
    for name in names:
        candidate = base_dir / name
        if candidate.is_dir():
            return candidate
    return None


def _find_existing_task_root(search_root: Path) -> Path | None:
    """Find the shallowest directory containing recognized task content."""
    candidates = [search_root]
    for candidate in sorted(search_root.rglob("*"), key=str):
        if not candidate.is_dir():
            continue
        try:
            depth = len(candidate.relative_to(search_root).parts)
        except ValueError:
            continue
        if depth <= _TASK_ROOT_SEARCH_DEPTH:
            candidates.append(candidate)

    for candidate in candidates:
        if _has_task_content(candidate):
            return candidate
    return None


def _find_supervisely_pairs(
    search_root: Path,
) -> list[tuple[Path, Path, str | None]]:
    """Locate sibling image/annotation folders in a raw Supervisely export."""
    pairs: list[tuple[Path, Path, str | None]] = []
    seen: set[tuple[str, str]] = set()

    for ann_dir in sorted(search_root.rglob("*"), key=str):
        if not ann_dir.is_dir() or ann_dir.name.lower() not in _ANN_DIR_NAMES:
            continue

        image_dir = None
        for image_name in _IMAGE_DIR_NAMES:
            candidate = ann_dir.parent / image_name
            if candidate.is_dir():
                image_dir = candidate
                break

        if image_dir is None:
            continue

        key = (str(image_dir.resolve()), str(ann_dir.resolve()))
        if key in seen:
            continue

        seen.add(key)
        pairs.append((image_dir, ann_dir, _infer_split_name(ann_dir.parent, search_root)))

    pairs.sort(key=lambda item: (item[2] or "zzz", str(item[0]), str(item[1])))
    return pairs


def _infer_split_name(path: Path, search_root: Path) -> str | None:
    """Infer train/val/test from the directory path when possible."""
    ancestors = [path]
    for parent in path.parents:
        ancestors.append(parent)
        if parent == search_root:
            break

    for candidate in ancestors:
        name = candidate.name.lower()
        if name in {"train", "training"}:
            return "train"
        if name in {"val", "valid", "validation"}:
            return "val"
        if name == "test":
            return "test"
    return None


def _collect_supervisely_items(
    pairs: list[tuple[Path, Path, str | None]],
) -> list[tuple[Path, Path, str | None]]:
    """Build a list of matching raw image/annotation pairs."""
    items: list[tuple[Path, Path, str | None]] = []

    for image_dir, ann_dir, split_name in pairs:
        image_index = _index_images(image_dir)
        for ann_file in sorted(ann_dir.rglob("*.json"), key=str):
            img_path = image_index.get(ann_file.stem.lower())
            if img_path is not None:
                items.append((img_path, ann_file, split_name))

    return items


def _index_images(image_dir: Path) -> dict[str, Path]:
    """Index image files by stem for a raw export directory."""
    image_index: dict[str, Path] = {}
    for img_path in sorted(image_dir.rglob("*"), key=str):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        image_index.setdefault(img_path.stem.lower(), img_path)
    return image_index


def _assign_detection_splits(
    items: list[tuple[Path, Path, str | None]],
) -> list[str]:
    """Map raw items onto train/val/test splits for YOLO conversion."""
    explicit_train = [idx for idx, (_, _, split) in enumerate(items) if split == "train"]
    explicit_val = [idx for idx, (_, _, split) in enumerate(items) if split == "val"]
    explicit_test = [idx for idx, (_, _, split) in enumerate(items) if split == "test"]

    assignments = ["train"] * len(items)

    if explicit_train and explicit_val:
        for idx in explicit_val:
            assignments[idx] = "val"
        for idx in explicit_test:
            assignments[idx] = "test"
        return assignments

    if explicit_train and explicit_test:
        for idx in explicit_test:
            assignments[idx] = "val"
        return assignments

    import numpy as np

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(items))
    split_idx = int(len(items) * 0.8)
    if len(items) > 1:
        split_idx = min(max(split_idx, 1), len(items) - 1)

    train_indices = set(indices[:split_idx].tolist())
    return ["train" if idx in train_indices else "val" for idx in range(len(items))]


def _unique_output_name(img_path: Path, used_names: set[str]) -> str:
    """Generate a stable, collision-free output filename."""
    candidate = img_path.name
    if candidate.lower() not in used_names:
        used_names.add(candidate.lower())
        return candidate

    candidate = f"{img_path.parent.name}_{img_path.stem}{img_path.suffix.lower()}"
    if candidate.lower() not in used_names:
        used_names.add(candidate.lower())
        return candidate

    counter = 1
    while True:
        candidate = f"{img_path.stem}_{counter}{img_path.suffix.lower()}"
        if candidate.lower() not in used_names:
            used_names.add(candidate.lower())
            return candidate
        counter += 1
