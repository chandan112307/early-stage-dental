<div align="center">

# 🦷 Early-Stage Dental Caries Detection

**AI-powered clinical decision-support system for early-stage dental caries detection from radiographic images**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Overview

This project delivers an end-to-end pipeline for **early-stage dental caries detection** from dental radiographs (X-rays). It combines three deep-learning models — **MobileNetV2** for classification, **YOLOv8** for object detection, and **U-Net** for semantic segmentation — into a unified inference pipeline served by a FastAPI backend and visualized through a React clinical dashboard.

The system enforces a **single source of truth** at every layer:

- **Training** — one centralized dataset pipeline, no bypass arguments
- **Backend** — real model inference only, no demo/synthetic fallbacks
- **Frontend** — all UI elements derived from backend API responses

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Clinical Dashboard                           │
│                     (React 18 · Port 3000)                          │
│  ┌──────────┐ ┌───────────┐ ┌────────────┐ ┌────────────────────┐  │
│  │ Sidebar  │ │  Header   │ │ X-Ray      │ │ Analysis Results   │  │
│  │          │ │           │ │ Viewer     │ │ Clinical Metrics   │  │
│  │          │ │           │ │ + Upload   │ │  (from /api)       │  │
│  └──────────┘ └───────────┘ └─────┬──────┘ └────────────────────┘  │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │  HTTP POST /api/predict
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                               │
│                     (Uvicorn · Port 8000)                            │
│                                                                     │
│  ┌──────────────┐   ┌────────────────────────────────────────────┐  │
│  │ Preprocessing │──▶│          Inference Pipeline               │  │
│  │  (CLAHE,      │   │  ┌──────────┐ ┌────────┐ ┌───────────┐  │  │
│  │   Resize,     │   │  │MobileNet │ │ YOLOv8 │ │   U-Net   │  │  │
│  │   Validate)   │   │  │Classifier│ │Detector│ │ Segmentor │  │  │
│  └──────────────┘   │  └──────────┘ └────────┘ └───────────┘  │  │
│                      └──────────────┬────────────────────────────┘  │
│  ┌──────────────┐                   │                               │
│  │Postprocessing│◀──────────────────┘                               │
│  │ (Annotate,   │                                                   │
│  │  Encode)     │                                                   │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Automated Training Pipeline                      │
│                         python -m training                           │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐  │
│  │ train_mobilenet │  │   train_yolo   │  │    train_unet        │  │
│  │ (TF/Keras)     │  │ (Ultralytics)  │  │  (Custom Keras)      │  │
│  └────────────────┘  └────────────────┘  └──────────────────────┘  │
│           │                   │                     │               │
│           ▼                   ▼                     ▼               │
│   ONNX export ──────── deploy to backend/models/ ──────▶ ready     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
early-stage-dental/
├── README.md
│
├── backend/                              # FastAPI inference backend
│   ├── main.py                           # Application entry point
│   ├── requirements.txt
│   ├── api/
│   │   └── routes.py                     # /predict, /health, /metrics endpoints
│   ├── configs/
│   │   └── config.py                     # Settings & model paths
│   ├── inference/
│   │   ├── pipeline.py                   # Inference orchestrator
│   │   ├── classifier.py                 # MobileNet classifier
│   │   ├── detector.py                   # YOLO detector
│   │   └── segmentor.py                  # U-Net segmentor
│   ├── model_loader/
│   │   └── loader.py                     # Model loader (fails if models missing)
│   ├── preprocessing/
│   │   └── preprocess.py                 # CLAHE enhancement & validation
│   ├── postprocessing/
│   │   └── postprocess.py                # Image annotation & encoding
│   ├── metrics/
│   │   └── metrics.py                    # Evaluation metrics endpoint
│   └── models/                           # ONNX model files (git-ignored)
│
├── frontend/                             # React clinical dashboard
│   ├── package.json
│   └── src/
│       ├── App.js                        # Root component & state
│       ├── components/
│       │   ├── Sidebar.js                # Navigation sidebar
│       │   ├── Header.js                 # Minimal top header
│       │   ├── XRayViewer.js             # X-ray viewer with bounding-box overlays
│       │   ├── UploadArea.js             # Drag-and-drop file upload
│       │   ├── AnalysisResults.js        # Findings & confidence from API
│       │   ├── ClinicalMetrics.js        # Metrics fetched from /api/metrics
│       │   └── LoadingState.js           # Processing animation overlay
│       └── utils/
│           └── api.js                    # HTTP client
│
└── training/                             # Automated training pipeline
    ├── __main__.py                       # End-to-end orchestrator
    ├── README.md
    ├── requirements.txt
    ├── configs/
    │   └── config.py                     # Hyperparameters & paths
    ├── data/
    │   ├── dataset.py                    # Dataset discovery & loading
    │   └── dataset_utils.py              # Centralized dataset pipeline
    ├── training/
    │   ├── train_mobilenet.py            # MobileNetV2 trainer
    │   ├── train_yolo.py                 # YOLOv8 trainer
    │   └── train_unet.py                 # U-Net trainer
    ├── export/
    │   ├── export_onnx.py                # ONNX conversion
    │   └── deploy.py                     # Deploy to backend/models/
    └── evaluation/
        └── evaluate.py                   # Metric computation
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| **Frontend** | React 18, CSS Custom Properties | Clinical dashboard UI |
| **Backend** | FastAPI, Uvicorn, Pydantic v2 | REST API & inference serving |
| **Image Processing** | OpenCV (headless), NumPy | CLAHE enhancement, resizing, annotation |
| **Classification** | TensorFlow / Keras — MobileNetV2 | Binary caries classification (224×224) |
| **Detection** | Ultralytics — YOLOv8 | Bounding-box caries localization (640×640) |
| **Segmentation** | Custom Keras — U-Net | Pixel-level caries mask (256×256) |
| **Model Format** | ONNX Runtime | Cross-framework inference |

---

## ✨ Features

- **Multi-model inference pipeline** — classification → detection → segmentation in a single request
- **Real-time clinical dashboard** — upload an X-ray and receive annotated results
- **Bounding-box overlays** — detected caries regions rendered on the radiograph
- **Segmentation masks** — pixel-level affected-area percentage
- **CLAHE preprocessing** — enhanced X-ray clarity
- **Confidence scoring** — per-prediction probability with visual indicators
- **Clinical metrics** — Accuracy, Precision, Recall, and F1-Score from real training data
- **Automated training pipeline** — `python -m training` handles everything end-to-end
- **No simulated outputs** — backend requires real trained models
- **Centralized dataset pipeline** — single source of truth for data acquisition

---

## 📋 Prerequisites

| Requirement | Version |
|:------------|:--------|
| Python | 3.10 or higher |
| Node.js | 16 or higher |
| npm | 8 or higher |

---

## 🚀 Quick Start

### 1. Train Models

```bash
# Install training dependencies
pip install -r training/requirements.txt

# Run the full pipeline (downloads data → trains → exports ONNX → deploys)
python -m training
```

This produces ONNX model files in `backend/models/`.

### 2. Start the Backend

```bash
pip install -r backend/requirements.txt

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend requires all three ONNX model files to be present. If models
are missing, it will refuse to start and instruct you to run the training
pipeline.

### 3. Start the Frontend

```bash
cd frontend
npm install
npm start
# Opens http://localhost:3000
```

### 4. Use the Application

1. Open **http://localhost:3000** in your browser.
2. Drag and drop a dental X-ray image onto the upload area.
3. View classification, detection, and segmentation results.

---

## 📡 API Documentation

Base URL: `http://localhost:8000`

### `POST /api/predict`

Upload a dental radiograph for multi-model inference.

**Request**

| Parameter | Type | Location | Description |
|:----------|:-----|:---------|:------------|
| `file` | `UploadFile` | Form-data | Dental X-ray image (`.jpg`, `.jpeg`, `.png`; max 10 MB) |

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@dental_xray.jpg"
```

**Response** `200 OK`

```json
{
  "prediction": "Caries",
  "confidence": 0.92,
  "probabilities": {
    "No Caries": 0.08,
    "Caries": 0.92
  },
  "processed_image_url": "/outputs/result_xxx.png",
  "annotated_base64": "base64...",
  "bounding_boxes": [
    {
      "x_min": 100,
      "y_min": 200,
      "x_max": 200,
      "y_max": 300,
      "label": "Caries",
      "confidence": 0.92
    }
  ],
  "segmentation_data": {
    "affected_percentage": 12.5,
    "mask_base64": "base64..."
  },
  "model_info": {
    "classifier_loaded": true,
    "detector_loaded": true,
    "segmentor_loaded": true
  },
  "inference_time_ms": 104.98
}
```

**Error Responses**

| Code | Description |
|:-----|:------------|
| `400` | Invalid file type or file exceeds size limit |
| `500` | Internal server error during inference |

---

### `GET /api/health`

```json
{
  "status": "ok",
  "models_loaded": true
}
```

---

### `GET /api/metrics`

Returns pre-computed evaluation metrics for all three models.

```json
{
  "classifier": {
    "accuracy": 0.94,
    "precision": 0.93,
    "recall": 0.95,
    "f1_score": 0.94
  },
  "detector": {
    "mAP50": 0.89,
    "mAP50_95": 0.72
  },
  "segmentor": {
    "dice_coefficient": 0.85,
    "iou": 0.82,
    "pixel_accuracy": 0.96
  }
}
```

Returns `503` if metrics are not available (training not yet completed).

---

## 🧠 Training Pipeline

A single command trains all models, exports to ONNX, and deploys:

```bash
python -m training
```

### Individual Models

Each module is directly executable and triggers the full dataset pipeline:

```bash
python -m training.training.train_mobilenet
python -m training.training.train_yolo
python -m training.training.train_unet
```

### Pipeline Options

```bash
python -m training --model mobilenet          # Train specific model
python -m training --model yolo unet          # Train multiple
python -m training --skip-export              # Train only
python -m training --epochs 100 --batch-size 8
```

### Models

- **MobileNetV2** — ImageNet-pretrained classification head, 224×224 input
- **YOLOv8** — Real-time object detection, 640×640 input
- **U-Net** — Encoder–decoder segmentation, 256×256 input

---

## ⚙️ Configuration

### Backend — `backend/configs/config.py`

| Setting | Default | Description |
|:--------|:--------|:------------|
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins |
| `SUPPORTED_EXTENSIONS` | `{".jpg", ".jpeg", ".png"}` | Accepted upload formats |
| `MAX_FILE_SIZE_MB` | `10` | Maximum upload size |
| `CLASSIFIER_MODEL_FILE` | `mobilenet_classifier.onnx` | Classification model |
| `DETECTOR_MODEL_FILE` | `yolo_detector.onnx` | Detection model |
| `SEGMENTOR_MODEL_FILE` | `unet_segmentor.onnx` | Segmentation model |

### Training — `training/configs/config.py`

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `BATCH_SIZE` | `16` | Training batch size |
| `EPOCHS` | `50` | Maximum training epochs |
| `LEARNING_RATE` | `1e-4` | Initial learning rate |
| `SEED` | `42` | Reproducibility seed |

---

## 🔒 Design Invariants

These properties are enforced structurally, not by convention:

1. **No execution path skips dataset preparation** — every training module calls `ensure_dataset(DATASET_DIR)` before any data access
2. **No dataset source ambiguity** — single download source, single validation function, no overrides
3. **No simulated outputs** — backend requires all model files present; missing models cause explicit failure
4. **No frontend elements exceed backend capability** — every UI element derives from real API responses
5. **All modules agree on dataset location and structure** — centralized in `training/configs/config.py`

---

## ♿ Accessibility

- **Semantic HTML** — `<main>`, `<nav>`, `<header>`, `<button>` used appropriately
- **ARIA attributes** — `role="alert"` on errors, `aria-label` on icon buttons
- **Keyboard navigation** — all interactive elements reachable via Tab
- **Focus management** — visible focus rings on interactive elements

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for dental professionals and AI researchers**

</div>
