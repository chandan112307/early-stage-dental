"""Model loader with singleton pattern.

All three ONNX model files must be present.  If any model is missing or
fails to load, the system raises an error — there is no demo mode or
synthetic fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.configs.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when one or more required models cannot be loaded."""


@dataclass
class LoadedModels:
    """Container for loaded model objects."""

    classifier: Any = None
    detector: Any = None
    segmentor: Any = None
    _loaded: bool = field(default=False, repr=False)


_models: LoadedModels | None = None


def _load_onnx(path: Path, label: str) -> Any:
    """Load an ONNX model file.  Raises on failure."""
    if not path.exists():
        raise ModelLoadError(
            f"{label} model not found at {path}. "
            f"Run the training pipeline first: python -m training"
        )
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        session = ort.InferenceSession(str(path))
        logger.info("%s model loaded from %s", label, path)
        return session
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load {label} model from {path}: {exc}"
        ) from exc


def load_models(settings: Settings | None = None) -> LoadedModels:
    """Load all models once (singleton). Safe to call multiple times.

    Raises
    ------
    ModelLoadError
        If any required model file is missing or fails to load.
    """
    global _models  # noqa: PLW0603
    if _models is not None and _models._loaded:
        return _models

    if settings is None:
        settings = get_settings()

    models = LoadedModels()
    models.classifier = _load_onnx(settings.classifier_path, "Classifier")
    models.detector = _load_onnx(settings.detector_path, "Detector")
    models.segmentor = _load_onnx(settings.segmentor_path, "Segmentor")

    logger.info("All models loaded — running in PRODUCTION mode")
    models._loaded = True
    _models = models
    return models


def get_models() -> LoadedModels:
    """Return the already-loaded models singleton."""
    if _models is None:
        return load_models()
    return _models


def reset_models() -> None:
    """Reset the singleton (useful for testing)."""
    global _models  # noqa: PLW0603
    _models = None
