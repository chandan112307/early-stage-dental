from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dataset(dataset_path: Path, kaggle_dataset: str) -> Path:
    # 1. Already exists
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"[INFO] Dataset found at {dataset_path}")
        return dataset_path

    # 2. Try DentalAI (dataset-tools)
    try:
        import dataset_tools as dtools

        print("[INFO] Downloading DentalAI dataset via dataset-tools …")
        dtools.download(dataset="Dentalai", dst_dir=str(dataset_path))

        print(f"[INFO] DentalAI downloaded to {dataset_path}")

        _convert_dentalai(dataset_path)

        return dataset_path

    except Exception as exc:
        print(f"[WARN] DentalAI download failed: {exc}")

    # 3. KaggleHub
    try:
        import kagglehub

        print(f"[INFO] Downloading dataset via kagglehub: {kaggle_dataset} …")
        downloaded = kagglehub.dataset_download(kaggle_dataset)
        return Path(downloaded)

    except Exception as exc:
        print(f"[WARN] kagglehub failed: {exc}")

    # 4. Kaggle CLI
    try:
        dataset_path.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                kaggle_dataset,
                "--unzip",
                "-p",
                str(dataset_path),
            ],
            check=True,
        )
        return dataset_path

    except Exception as exc:
        print(f"[WARN] Kaggle CLI failed: {exc}")

    print("[ERROR] Dataset download failed")
    sys.exit(1)


# ------------------------------------------------------------
# 🔥 DentalAI Conversion (WORKING BASE VERSION)
# ------------------------------------------------------------

def _convert_dentalai(dataset_path: Path):
    """
    Converts DentalAI Supervisely format into:
    - classification/
    - detection/
    - segmentation/
    """

    if (dataset_path / "classification").exists():
        print("[INFO] Dataset already converted")
        return

    print("[INFO] Converting DentalAI dataset...")

    images_dir = list(dataset_path.rglob("img"))[0]
    ann_dir = list(dataset_path.rglob("ann"))[0]

    # Create folders
    cls_dir = dataset_path / "classification"
    det_img_dir = dataset_path / "detection/images"
    det_lbl_dir = dataset_path / "detection/labels"
    seg_img_dir = dataset_path / "segmentation/images"
    seg_mask_dir = dataset_path / "segmentation/masks"

    for d in [cls_dir, det_img_dir, det_lbl_dir, seg_img_dir, seg_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    (cls_dir / "caries").mkdir(exist_ok=True)
    (cls_dir / "no_caries").mkdir(exist_ok=True)

    class_map = {"caries": 0}

    for ann_file in ann_dir.glob("*.json"):
        with open(ann_file) as f:
            ann = json.load(f)

        img_name = ann_file.stem + ".jpg"
        img_path = images_dir / img_name

        if not img_path.exists():
            continue

        has_caries = False

        # Copy image to segmentation + detection
        shutil.copy(img_path, seg_img_dir / img_name)
        shutil.copy(img_path, det_img_dir / img_name)

        # Create mask
        import cv2
        import numpy as np

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        yolo_labels = []

        for obj in ann.get("objects", []):
            label = obj.get("classTitle", "").lower()

            if "caries" in label:
                has_caries = True

                points = obj.get("points", {}).get("exterior", [])

                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]

                    # YOLO format
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = abs(x2 - x1) / w
                    bh = abs(y2 - y1) / h

                    yolo_labels.append(f"0 {xc} {yc} {bw} {bh}")

                    # Mask
                    mask[int(y1):int(y2), int(x1):int(x2)] = 255

        # Save YOLO label
        with open(det_lbl_dir / (ann_file.stem + ".txt"), "w") as f:
            f.write("\n".join(yolo_labels))

        # Save mask
        cv2.imwrite(str(seg_mask_dir / img_name), mask)

        # Classification copy
        if has_caries:
            shutil.copy(img_path, cls_dir / "caries" / img_name)
        else:
            shutil.copy(img_path, cls_dir / "no_caries" / img_name)

    # Create YOLO data.yaml
    yaml_content = f"""
train: {det_img_dir}
val: {det_img_dir}
nc: 1
names: ['caries']
"""

    with open(dataset_path / "detection/data.yaml", "w") as f:
        f.write(yaml_content)

    print("[INFO] Conversion complete")
