"""Automatic dataset download utility for the training pipeline.

Provides :func:`ensure_dataset` which guarantees a dataset directory is
available before training begins.  It first checks for a local path; if
the directory is missing or empty it attempts to download from Kaggle
using *kagglehub* (primary) and the *Kaggle CLI* (fallback).

No Kaggle credentials are embedded – the caller is expected to have
``kaggle.json`` configured (e.g. uploaded in Google Colab).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dataset(dataset_path: Path, kaggle_dataset: str) -> Path:
    """Return *dataset_path* after ensuring it contains data.

    Resolution order:

    1. If *dataset_path* already exists **and** is non-empty, return it
       immediately.
    2. Try downloading via ``kagglehub.dataset_download``.
    3. Fall back to the ``kaggle`` CLI tool.
    4. If both fail, print a clear error message and exit.

    Parameters
    ----------
    dataset_path:
        Local directory where the dataset should reside.
    kaggle_dataset:
        Kaggle dataset identifier, e.g. ``"owner/dataset-name"``.

    Returns
    -------
    Path
        The directory containing the downloaded (or already-present) dataset.
    """
    # 1. Already available locally?
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"[INFO] Dataset found at {dataset_path}")
        return dataset_path

    # 2. Try kagglehub --------------------------------------------------
    try:
        import kagglehub  # type: ignore[import-untyped]

        print(f"[INFO] Downloading dataset via kagglehub: {kaggle_dataset} …")
        downloaded = kagglehub.dataset_download(kaggle_dataset)
        downloaded_path = Path(downloaded)
        print(f"[INFO] Dataset downloaded via kagglehub to {downloaded_path}")
        return downloaded_path
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] kagglehub download failed: {exc}")

    # 3. Fallback to Kaggle CLI -----------------------------------------
    try:
        dataset_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Downloading dataset via Kaggle CLI: {kaggle_dataset} …")
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
        print(f"[INFO] Dataset downloaded via Kaggle CLI to {dataset_path}")
        return dataset_path
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Kaggle CLI download failed: {exc}")

    # 4. Nothing worked --------------------------------------------------
    print(
        "\n[ERROR] Could not download the dataset automatically.\n"
        "Please do one of the following:\n"
        "  • Provide the --dataset /path/to/dataset argument\n"
        "  • Place your kaggle.json credentials and retry\n"
        "  • Manually download the dataset and pass the path\n"
    )
    sys.exit(1)
