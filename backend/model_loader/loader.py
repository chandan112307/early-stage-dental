"""Model loader with singleton pattern and graceful demo fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.configs.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class LoadedModels:
    """Container for loaded model objects."""

    classifier: Any | None = None
    detector: Any | None = None
    segmentor: Any | None = None
    demo_mode: bool = True
    _loaded: bool = field(default=False, repr=False)


_models: LoadedModels | None = None


def _try_load_onnx(path: Path, label: str) -> Any | None:
    """Attempt to load an ONNX model file. Returns None on failure."""
    if not path.exists():
        logger.info("%s model not found at %s – using demo mode", label, path)
        return None
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        session = ort.InferenceSession(str(path))
        logger.info("%s model loaded from %s", label, path)
        return session
    except Exception:
        logger.warning(
            "Failed to load %s model from %s – falling back to demo mode",
            label,
            path,
            exc_info=True,
        )
        return None


def load_models(settings: Settings | None = None) -> LoadedModels:
    """Load all models once (singleton). Safe to call multiple times."""
    global _models  # noqa: PLW0603
    if _models is not None and _models._loaded:
        return _models

    if settings is None:
        settings = get_settings()

    models = LoadedModels()

    if not settings.DEMO_MODE:
        models.classifier = _try_load_onnx(
            settings.classifier_path, "Classifier"
        )
        models.detector = _try_load_onnx(settings.detector_path, "Detector")
        models.segmentor = _try_load_onnx(
            settings.segmentor_path, "Segmentor"
        )
        models.demo_mode = any(
            m is None
            for m in [models.classifier, models.detector, models.segmentor]
        )
    else:
        models.demo_mode = True

    if models.demo_mode:
        logger.info("Running in DEMO mode – returning synthetic results")
    else:
        logger.info("All models loaded – running in PRODUCTION mode")

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
