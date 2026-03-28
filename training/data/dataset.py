"""Dataset loading and splitting for dental caries classification.

Supports either of these directory layouts::

    dataset_root/
        classification/
            no_caries/
            caries/

or::

    dataset_root/
        No Caries/
        Caries/

The :class:`DentalDataset` class handles discovery, splitting, and
batch generation for TensorFlow / Keras training loops. It also tolerates
nested image folders inside each class directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from training.configs.config import (
    CLASS_NAMES,
    MOBILENET_IMG_SIZE,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from training.preprocessing.preprocess import (
    apply_clahe,
    normalize_image,
    resize_image,
)

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_CLASS_NAME_ALIASES = {
    "caries": ("caries", "Caries"),
    "no caries": ("no_caries", "No Caries"),
}


class DentalDataset:
    """Discovers, loads, and splits a dental X-ray image dataset.

    Parameters
    ----------
    root_dir:
        Path to the dataset root directory.
    class_names:
        Ordered list of class-folder names.
    target_size:
        ``(width, height)`` for image resizing.
    train_ratio:
        Proportion of data for training.
    val_ratio:
        Proportion of data for validation.
    test_ratio:
        Proportion of data for testing.
    seed:
        Random seed for reproducible splits.
    apply_clahe:
        Whether to apply CLAHE during loading.
    """

    def __init__(
        self,
        root_dir: str | Path,
        class_names: Optional[List[str]] = None,
        target_size: Tuple[int, int] = MOBILENET_IMG_SIZE,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        seed: int = SEED,
        apply_clahe: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.class_names = class_names or list(CLASS_NAMES)
        self.target_size = target_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.apply_clahe = apply_clahe

        self._image_paths: List[str] = []
        self._labels: List[int] = []
        self._discover_images()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def num_samples(self) -> int:
        """Total number of discovered images."""
        return len(self._image_paths)

    @property
    def class_distribution(self) -> Dict[str, int]:
        """Per-class sample counts."""
        dist: Dict[str, int] = {name: 0 for name in self.class_names}
        for label in self._labels:
            dist[self.class_names[label]] += 1
        return dist

    def split(
        self,
    ) -> Tuple[
        Tuple[List[str], List[int]],
        Tuple[List[str], List[int]],
        Tuple[List[str], List[int]],
    ]:
        """Split the dataset into train / val / test sets.

        Returns
        -------
        tuple
            ``((train_paths, train_labels),
              (val_paths, val_labels),
              (test_paths, test_labels))``
        """
        if self.num_samples == 0:
            raise ValueError(
                f"No images found in {self.root_dir}. Ensure subdirectories "
                f"match class names: {self.class_names}"
            )

        paths = np.array(self._image_paths)
        labels = np.array(self._labels)

        val_test_ratio = self.val_ratio + self.test_ratio
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths,
            labels,
            test_size=val_test_ratio,
            random_state=self.seed,
            stratify=labels,
        )

        relative_test = self.test_ratio / val_test_ratio
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths,
            temp_labels,
            test_size=relative_test,
            random_state=self.seed,
            stratify=temp_labels,
        )

        return (
            (train_paths.tolist(), train_labels.tolist()),
            (val_paths.tolist(), val_labels.tolist()),
            (test_paths.tolist(), test_labels.tolist()),
        )

    def load_images(
        self,
        paths: List[str],
        labels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a list of images into NumPy arrays.

        Parameters
        ----------
        paths:
            File paths to load.
        labels:
            Corresponding integer labels.

        Returns
        -------
        tuple
            ``(images, labels)`` as NumPy arrays.
        """
        images: List[np.ndarray] = []
        valid_labels: List[int] = []

        for path, label in zip(paths, labels):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if self.apply_clahe:
                img = apply_clahe(img)
            img = resize_image(img, self.target_size)
            img = normalize_image(img)
            images.append(img)
            valid_labels.append(label)

        return np.array(images, dtype=np.float32), np.array(
            valid_labels, dtype=np.int32
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_images(self) -> None:
        """Walk the root directory and collect image paths with labels."""
        base_dir = self._resolve_base_dir()
        for label_idx, class_name in enumerate(self.class_names):
            class_dir = self._resolve_class_dir(base_dir, class_name)
            if not class_dir.is_dir():
                continue
            for entry in sorted(class_dir.rglob("*"), key=str):
                if entry.is_file() and entry.suffix.lower() in _SUPPORTED_EXTENSIONS:
                    self._image_paths.append(str(entry))
                    self._labels.append(label_idx)

    def _resolve_base_dir(self) -> Path:
        """Prefer the canonical classification subdirectory when present."""
        classification_dir = self.root_dir / "classification"
        if classification_dir.is_dir():
            return classification_dir
        return self.root_dir

    def _resolve_class_dir(self, base_dir: Path, class_name: str) -> Path:
        """Locate a class directory using a small alias set."""
        for candidate_name in _candidate_class_dir_names(class_name):
            candidate = base_dir / candidate_name
            if candidate.is_dir():
                return candidate
        return base_dir / class_name


def _candidate_class_dir_names(class_name: str) -> Tuple[str, ...]:
    """Return likely directory names for a semantic class label."""
    normalized = class_name.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    aliases = _CLASS_NAME_ALIASES.get(normalized, ())
    ordered = dict.fromkeys((class_name, *aliases))
    return tuple(ordered.keys())
