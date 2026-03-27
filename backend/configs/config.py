"""Backend configuration."""

import os
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings with sensible defaults."""

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "models"
    OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"

    # File validation
    SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
    MAX_FILE_SIZE_MB: int = 10
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024

    # Model input sizes
    MOBILENET_SIZE: tuple[int, int] = (224, 224)
    YOLO_SIZE: tuple[int, int] = (640, 640)
    UNET_SIZE: tuple[int, int] = (256, 256)

    # Model file names
    CLASSIFIER_MODEL_FILE: str = "mobilenet_classifier.onnx"
    DETECTOR_MODEL_FILE: str = "yolo_detector.onnx"
    SEGMENTOR_MODEL_FILE: str = "unet_segmentor.onnx"

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Demo mode: auto-detected if model files are missing
    DEMO_MODE: bool = True

    @property
    def classifier_path(self) -> Path:
        return self.MODEL_DIR / self.CLASSIFIER_MODEL_FILE

    @property
    def detector_path(self) -> Path:
        return self.MODEL_DIR / self.DETECTOR_MODEL_FILE

    @property
    def segmentor_path(self) -> Path:
        return self.MODEL_DIR / self.SEGMENTOR_MODEL_FILE

    def detect_demo_mode(self) -> bool:
        """Return True if any model file is missing."""
        return not all(
            p.exists()
            for p in [
                self.classifier_path,
                self.detector_path,
                self.segmentor_path,
            ]
        )


def get_settings() -> Settings:
    """Create settings and auto-detect demo mode."""
    settings = Settings()

    # Allow override via environment variable
    override = os.getenv("DEMO_MODE")
    if override is not None:
        settings.DEMO_MODE = override.lower() in ("1", "true", "yes")
    else:
        settings.DEMO_MODE = settings.detect_demo_mode()

    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return settings
