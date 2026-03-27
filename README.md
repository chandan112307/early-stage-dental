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

The system is designed for dental professionals and researchers who need a rapid, interpretable second opinion on radiographic findings. When trained models are not available, the application automatically falls back to a **demo mode** that generates synthetic results, making it easy to explore the full interface without GPU hardware or training data.

### Screenshot

![Clinical Dashboard](https://github.com/user-attachments/assets/bde1634f-3334-40ab-9a1d-a1d3e84323a3)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Clinical Dashboard                           │
│                     (React 18 · Port 3000)                          │
│  ┌──────────┐ ┌───────────┐ ┌────────────┐ ┌────────────────────┐  │
│  │ Sidebar  │ │  Header   │ │ X-Ray      │ │ Analysis Results   │  │
│  │          │ │           │ │ Viewer     │ │ Clinical Metrics   │  │
│  │          │ │           │ │ + Upload   │ │ Diagnostic History │  │
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
│                     Offline Training Pipeline                        │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐  │
│  │ train_mobilenet │  │   train_yolo   │  │    train_unet        │  │
│  │ (TF/Keras)     │  │ (Ultralytics)  │  │  (Custom Keras)      │  │
│  └────────────────┘  └────────────────┘  └──────────────────────┘  │
│           │                   │                     │               │
│           ▼                   ▼                     ▼               │
│     .onnx model         .onnx model           .onnx model          │
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
│   ├── requirements.txt                  # Python dependencies
│   ├── __init__.py
│   ├── api/
│   │   ├── routes.py                     # /predict, /health, /metrics endpoints
│   │   └── __init__.py
│   ├── configs/
│   │   ├── config.py                     # Settings & model paths
│   │   └── __init__.py
│   ├── inference/
│   │   ├── pipeline.py                   # Inference orchestrator
│   │   ├── classifier.py                 # MobileNet classifier (real / demo)
│   │   ├── detector.py                   # YOLO detector (real / demo)
│   │   ├── segmentor.py                  # U-Net segmentor (real / demo)
│   │   └── __init__.py
│   ├── model_loader/
│   │   ├── loader.py                     # Singleton model loader
│   │   └── __init__.py
│   ├── preprocessing/
│   │   ├── preprocess.py                 # CLAHE enhancement & validation
│   │   └── __init__.py
│   ├── postprocessing/
│   │   ├── postprocess.py                # Image annotation & encoding
│   │   └── __init__.py
│   ├── metrics/
│   │   ├── metrics.py                    # Evaluation metrics endpoint logic
│   │   └── __init__.py
│   ├── models/                           # ONNX model files (git-ignored)
│   └── outputs/                          # Processed result images
│
├── frontend/                             # React clinical dashboard
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── index.js                      # React entry point
│       ├── App.js                        # Root component & state management
│       ├── App.css                       # Main layout styles
│       ├── components/
│       │   ├── Sidebar.js / .css         # Navigation sidebar
│       │   ├── Header.js / .css          # Top header bar
│       │   ├── XRayViewer.js / .css      # X-ray viewer with bounding-box overlays
│       │   ├── UploadArea.js / .css      # Drag-and-drop file upload
│       │   ├── AnalysisResults.js / .css # Findings & confidence display
│       │   ├── ClinicalMetrics.js / .css # Accuracy, Precision, Recall, F1
│       │   ├── DiagnosticHistory.js/.css # Patient history table
│       │   └── LoadingState.js / .css    # Processing animation overlay
│       ├── styles/
│       │   └── variables.css             # CSS custom properties (design tokens)
│       └── utils/
│           └── api.js                    # HTTP client (predictImage, getHealth, getMetrics)
│
└── training/                             # Offline training pipeline
    ├── README.md                         # Training-specific docs
    ├── requirements.txt                  # Training Python dependencies
    ├── __init__.py
    ├── configs/
    │   ├── config.py                     # Hyperparameters & data-split ratios
    │   └── __init__.py
    ├── data/
    │   ├── dataset.py                    # Dataset discovery & loading
    │   ├── augmentation.py               # X-ray augmentation transforms
    │   └── __init__.py
    ├── preprocessing/
    │   ├── preprocess.py                 # CLAHE & normalization
    │   └── __init__.py
    ├── training/
    │   ├── train_mobilenet.py            # MobileNetV2 classification trainer
    │   ├── train_yolo.py                 # YOLOv8 detection trainer
    │   ├── train_unet.py                 # U-Net segmentation trainer
    │   └── __init__.py
    ├── evaluation/
    │   ├── evaluate.py                   # Metric computation & reporting
    │   └── __init__.py
    ├── models/                           # Saved checkpoints (git-ignored)
    ├── outputs/                          # TensorBoard logs & artifacts
    └── metrics/                          # Evaluation JSON reports
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| **Frontend** | React 18, CSS Custom Properties, Inter font | Clinical dashboard UI |
| **Backend** | FastAPI, Uvicorn, Pydantic v2 | REST API & inference serving |
| **Image Processing** | OpenCV (headless), Pillow, NumPy | CLAHE enhancement, resizing, annotation |
| **Classification** | TensorFlow / Keras — MobileNetV2 | Binary caries classification (224×224) |
| **Detection** | Ultralytics — YOLOv8 | Bounding-box caries localization (640×640) |
| **Segmentation** | Custom Keras — U-Net | Pixel-level caries mask (256×256) |
| **Model Format** | ONNX Runtime | Cross-framework inference |
| **Styling** | CSS custom properties (design tokens) | Consistent theming, responsive layout |

---

## ✨ Features

- **Multi-model inference pipeline** — classification → detection → segmentation in a single request
- **Real-time clinical dashboard** — upload an X-ray and receive annotated results instantly
- **Bounding-box overlays** — detected caries regions rendered directly on the radiograph
- **Segmentation masks** — pixel-level affected-area percentage calculation
- **CLAHE preprocessing** — Contrast-Limited Adaptive Histogram Equalization for enhanced X-ray clarity
- **Confidence scoring** — per-prediction probability with visual indicators
- **Clinical metrics panel** — Accuracy, Precision, Recall, and F1-Score at a glance
- **Drag-and-drop upload** — intuitive file selection supporting JPG, JPEG, and PNG (up to 10 MB)
- **Demo mode** — fully functional UI with synthetic results when trained models are unavailable
- **Responsive design** — adapts from desktop to tablet with collapsible sidebar
- **Accessible interface** — ARIA roles, keyboard navigation, semantic HTML

---

## 📋 Prerequisites

| Requirement | Version |
|:------------|:--------|
| Python | 3.10 or higher |
| Node.js | 16 or higher |
| npm | 8 or higher |
| Git | 2.x |

> **Note:** GPU hardware is **not required**. The backend runs in demo mode when ONNX model files are absent. To run real inference, place trained `.onnx` model files in `backend/models/`.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<owner>/early-stage-dental.git
cd early-stage-dental
```

### 2. Start the Backend

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r backend/requirements.txt

# Start the server (default: http://localhost:8000)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend starts in **demo mode** automatically if no model files are found in `backend/models/`.

### 3. Start the Frontend

```bash
cd frontend
npm install
npm start
# Opens http://localhost:3000
```

### 4. Use the Application

1. Open **http://localhost:3000** in your browser.
2. Drag and drop a dental X-ray image onto the upload area (or click to browse).
3. View classification results, bounding-box detections, and segmentation data in the results panel.

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
    "demo_mode": true,
    "classifier_loaded": false,
    "detector_loaded": false,
    "segmentor_loaded": false
  },
  "inference_time_ms": 104.98
}
```

**Response Fields**

| Field | Type | Description |
|:------|:-----|:------------|
| `prediction` | `string` | `"Caries"` or `"No Caries"` |
| `confidence` | `float` | Highest class probability (0.0 – 1.0) |
| `probabilities` | `object` | Per-class probability map |
| `processed_image_url` | `string` | Path to annotated output image |
| `annotated_base64` | `string` | Base64-encoded annotated image |
| `bounding_boxes` | `array` | Detected caries regions with coordinates and confidence |
| `segmentation_data` | `object` | `affected_percentage` and base64-encoded binary mask |
| `model_info` | `object` | Flags indicating which models are loaded and demo-mode status |
| `inference_time_ms` | `float` | Total pipeline inference time in milliseconds |

**Error Responses**

| Code | Description |
|:-----|:------------|
| `400` | Invalid file type or file exceeds size limit |
| `422` | Missing `file` field in form data |
| `500` | Internal server error during inference |

---

### `GET /api/health`

Health-check endpoint for monitoring and load balancers.

```bash
curl http://localhost:8000/api/health
```

**Response** `200 OK`

```json
{
  "status": "ok",
  "demo_mode": true,
  "models": {
    "classifier_loaded": false,
    "detector_loaded": false,
    "segmentor_loaded": false
  }
}
```

---

### `GET /api/metrics`

Returns pre-computed evaluation metrics for all three models.

```bash
curl http://localhost:8000/api/metrics
```

**Response** `200 OK`

```json
{
  "classification": {
    "accuracy": 0.94,
    "precision": 0.93,
    "recall": 0.95,
    "f1_score": 0.94
  },
  "detection": {
    "mAP50": 0.89,
    "mAP50_95": 0.72
  },
  "segmentation": {
    "mean_iou": 0.82,
    "dice_coefficient": 0.85
  }
}
```

---

## 🧠 Training Pipeline

The `training/` directory contains standalone scripts for training each model from scratch.

### MobileNetV2 — Classification

```bash
python -m training.training.train_mobilenet \
    --data-dir /path/to/dataset \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 0.0001
```

- **Architecture:** MobileNetV2 backbone (ImageNet pre-trained) + custom classification head with dropout
- **Input:** 224×224 RGB images
- **Output:** Binary — `No Caries` / `Caries`
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

### YOLOv8 — Object Detection

```bash
python -m training.training.train_yolo \
    --data /path/to/data.yaml \
    --epochs 100 \
    --batch-size 16
```

- **Architecture:** YOLOv8 (Ultralytics) — configurable size (nano / small / medium / large)
- **Input:** 640×640 images
- **Output:** Bounding boxes with class labels and confidence scores
- **Dataset:** YOLO-format (`images/`, `labels/`, `data.yaml`)

### U-Net — Segmentation

```bash
python -m training.training.train_unet \
    --image-dir /path/to/images \
    --mask-dir /path/to/masks \
    --epochs 50 \
    --batch-size 8
```

- **Architecture:** Encoder–decoder with skip connections (64→128→256→512 / bottleneck 1024)
- **Input:** 256×256 grayscale images
- **Output:** Binary segmentation mask (caries vs. background)
- **Loss:** Binary cross-entropy · **Metrics:** Accuracy, Mean IoU

### Data Augmentation

Augmentations applied during training (configurable in `training/configs/config.py`):

| Augmentation | Range |
|:-------------|:------|
| Rotation | ±15° |
| Brightness | 0.8 – 1.2 |
| Contrast | 0.8 – 1.2 |
| Horizontal flip | 50% probability |

### Data Splits

| Split | Ratio |
|:------|:------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

---

## ⚙️ Configuration Reference

### Backend — `backend/configs/config.py`

| Setting | Default | Description |
|:--------|:--------|:------------|
| `CORS_ORIGINS` | `["http://localhost:3000", "http://127.0.0.1:3000"]` | Allowed CORS origins |
| `SUPPORTED_EXTENSIONS` | `{".jpg", ".jpeg", ".png"}` | Accepted upload formats |
| `MAX_FILE_SIZE_MB` | `10` | Maximum upload size in megabytes |
| `MOBILENET_SIZE` | `(224, 224)` | MobileNet input resolution |
| `YOLO_SIZE` | `(640, 640)` | YOLO input resolution |
| `UNET_SIZE` | `(256, 256)` | U-Net input resolution |
| `CLASSIFIER_MODEL_FILE` | `mobilenet_classifier.onnx` | Classification model filename |
| `DETECTOR_MODEL_FILE` | `yolo_detector.onnx` | Detection model filename |
| `SEGMENTOR_MODEL_FILE` | `unet_segmentor.onnx` | Segmentation model filename |
| `DEMO_MODE` | Auto-detected | `True` if any model file is missing |

### Environment Variables

| Variable | Values | Description |
|:---------|:-------|:------------|
| `DEMO_MODE` | `true` / `false` / `1` / `0` / `yes` / `no` | Override automatic demo-mode detection |

```bash
# Force demo mode on
export DEMO_MODE=true

# Force demo mode off (requires model files)
export DEMO_MODE=false
```

### Training — `training/configs/config.py`

| Hyperparameter | Default | Description |
|:---------------|:--------|:------------|
| `BATCH_SIZE` | `16` | Training batch size |
| `EPOCHS` | `50` | Maximum training epochs |
| `LEARNING_RATE` | `1e-4` | Initial learning rate |
| `EARLY_STOPPING_PATIENCE` | `7` | Epochs without improvement before stopping |
| `CLAHE_CLIP_LIMIT` | `2.0` | CLAHE contrast clip limit |
| `CLAHE_TILE_GRID` | `(8, 8)` | CLAHE tile grid size |
| `RANDOM_SEED` | `42` | Reproducibility seed |
| `CLASSES` | `["No Caries", "Caries"]` | Classification labels |

---

## 🔄 Pipeline Flow

When an image is submitted to `POST /api/predict`, the backend executes the following sequential pipeline:

```
 Upload Image
      │
      ▼
┌─────────────────┐
│  Preprocessing   │  Validate format & size → CLAHE enhancement → Resize
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Classification  │  MobileNetV2 → "Caries" / "No Caries" + confidence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Detection      │  YOLOv8 → bounding boxes [{x_min, y_min, x_max, y_max, label, confidence}]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Segmentation    │  U-Net → binary mask + affected_percentage
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Postprocessing   │  Annotate image (boxes, mask overlay) → encode base64
└────────┬────────┘
         │
         ▼
   JSON Response
```

Each stage runs independently on the preprocessed image. Results from all three models are aggregated into a single response.

---

## 🖥️ UI Components

| Component | File | Description |
|:----------|:-----|:------------|
| **Sidebar** | `Sidebar.js` | Vertical navigation with four items: Overview, Upload, Reports, Settings. Collapsible on smaller screens. |
| **Header** | `Header.js` | Top bar displaying patient ID field, search, notifications icon, and user profile. |
| **XRayViewer** | `XRayViewer.js` | Central image display area. Renders the uploaded X-ray with bounding-box overlays drawn from prediction data. Supports zoom and rotation controls. |
| **UploadArea** | `UploadArea.js` | Drag-and-drop zone with click-to-browse fallback. Validates file type and size before upload. Displays preview thumbnail. |
| **AnalysisResults** | `AnalysisResults.js` | Primary findings card showing prediction label, confidence bar, inference time, and per-class probabilities. |
| **ClinicalMetrics** | `ClinicalMetrics.js` | Four-metric grid displaying Accuracy, Precision, Recall, and F1-Score fetched from `/api/metrics`. |
| **DiagnosticHistory** | `DiagnosticHistory.js` | Table of recent diagnostic records with patient ID, date, finding, and status columns. |
| **LoadingState** | `LoadingState.js` | Full-overlay spinner with pulsing animation shown during inference processing. |

---

## 🧑‍💻 Development Guide

### Backend Development

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run with auto-reload
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Access interactive API docs
open http://localhost:8000/docs        # Swagger UI
open http://localhost:8000/redoc       # ReDoc
```

### Frontend Development

```bash
cd frontend
npm install
npm start                              # Dev server at http://localhost:3000
npm run build                          # Production build → frontend/build/
npm test                               # Run test suite
```

### API Client (`frontend/src/utils/api.js`)

The frontend communicates with the backend through three functions:

```javascript
import { predictImage, getHealth, getMetrics } from './utils/api';

// Upload an image for analysis
const result = await predictImage(file);   // POST /api/predict

// Check backend status
const health = await getHealth();          // GET  /api/health

// Fetch model performance metrics
const metrics = await getMetrics();        // GET  /api/metrics
```

The base URL defaults to `http://localhost:8000`.

---

## 🎭 Demo Mode

The application ships with a **built-in demo mode** for development and demonstration purposes.

### How It Works

- On startup, the backend checks for ONNX model files in `backend/models/`.
- If **any** model file is missing, demo mode activates automatically.
- Demo mode can also be forced via the `DEMO_MODE` environment variable.

### Demo Behavior

| Model | Demo Output |
|:------|:------------|
| **Classifier** | Random prediction weighted 70% Caries / 30% No Caries |
| **Detector** | 1–3 synthetic bounding boxes at plausible dental coordinates |
| **Segmentor** | Synthetic binary mask with computed `affected_percentage` |

### Identifying Demo Mode

The `model_info` field in every `/api/predict` response indicates the current mode:

```json
{
  "model_info": {
    "demo_mode": true,
    "classifier_loaded": false,
    "detector_loaded": false,
    "segmentor_loaded": false
  }
}
```

The `/api/health` endpoint also reports `"demo_mode": true`.

---

## 📱 Responsive Design

The dashboard uses a fluid layout with CSS Grid and Flexbox:

| Breakpoint | Layout |
|:-----------|:-------|
| **≥ 1024 px** | Full layout — fixed sidebar, two-column content grid (`1fr 380px`) |
| **< 1024 px** | Collapsible sidebar (hamburger toggle), single-column stacked layout |

Key responsive behaviors:
- Sidebar collapses into an overlay drawer on narrow viewports
- Content grid shifts to a single column stack
- Upload area and X-ray viewer scale proportionally
- Typography and spacing adjust via CSS custom properties

### Design Tokens (`frontend/src/styles/variables.css`)

```css
:root {
  --color-primary:        #0D9488;    /* Teal */
  --color-success:        #10B981;    /* Green */
  --color-danger:         #EF4444;    /* Red */
  --color-warning:        #F59E0B;    /* Amber */
  --color-bg:             #F6F8FA;    /* Page background */
  --color-card:           #FFFFFF;    /* Card surfaces */
  --color-viewer-bg:      #111827;    /* X-ray viewer background */
  --color-text:           #1F2937;    /* Primary text */
  --color-text-secondary: #6B7280;    /* Secondary text */
  --color-border:         #E5E7EB;    /* Borders & dividers */
  --shadow-sm / --shadow-md / --shadow-lg;     /* Elevation */
  --radius: 12px;  --radius-sm: 8px;  --radius-xs: 6px;
  --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
```

---

## ♿ Accessibility

The interface follows WCAG guidelines where applicable:

- **Semantic HTML** — `<main>`, `<nav>`, `<header>`, `<button>`, `<table>` elements used appropriately
- **ARIA attributes** — `role="alert"` on error banners, `aria-label` on icon-only buttons
- **Keyboard navigation** — all interactive elements reachable via Tab; Enter/Space to activate
- **Color contrast** — text and UI elements meet AA contrast ratios against their backgrounds
- **Focus management** — visible focus rings on interactive elements
- **Screen-reader support** — meaningful alt text and status announcements

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for dental professionals and AI researchers**

</div>
