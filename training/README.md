# Dental Caries Detection – Training Pipeline

Automated training pipeline for early-stage dental caries detection.
Produces ONNX models consumed by the backend inference server.

## Architecture

```
training/
├── __main__.py                # End-to-end pipeline orchestrator
├── configs/
│   └── config.py              # All hyper-parameters, paths, and constants
├── data/
│   ├── dataset.py             # Dataset discovery, loading, and splitting
│   └── dataset_utils.py       # Centralized dataset pipeline (download + convert + validate)
├── preprocessing/
│   └── preprocess.py          # CLAHE enhancement, resize, normalise
├── training/
│   ├── train_mobilenet.py     # MobileNetV2 classification (TensorFlow)
│   ├── train_yolo.py          # YOLOv8 detection (Ultralytics)
│   └── train_unet.py          # U-Net segmentation (TensorFlow)
├── evaluation/
│   └── evaluate.py            # Metrics: accuracy, precision, recall, F1, IoU
├── export/
│   ├── export_onnx.py         # Convert trained models to ONNX
│   └── deploy.py              # Deploy ONNX models + metrics to backend/
├── models/                    # Saved model checkpoints (git-ignored)
├── outputs/                   # TensorBoard logs and run artefacts
├── metrics/                   # Evaluation JSON reports
└── requirements.txt           # Python dependencies
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r training/requirements.txt

# 2. Run the full pipeline (dataset → train → export → deploy)
python -m training
```

That single command:
1. Downloads and validates the dataset automatically
2. Trains all three models (MobileNet, YOLO, U-Net)
3. Exports trained models to ONNX format
4. Deploys ONNX models to `backend/models/`
5. Aggregates evaluation metrics for the backend

## Running Individual Models

Each training module is directly executable and triggers the full
dataset pipeline internally — no external orchestrator required:

```bash
python -m training.training.train_mobilenet
python -m training.training.train_yolo
python -m training.training.train_unet
```

Every module calls `ensure_dataset(DATASET_DIR)` before accessing data.
There are no dataset path arguments — the dataset source is singular
and final.

## Pipeline Options

```bash
# Train specific model(s) only
python -m training --model mobilenet
python -m training --model yolo unet

# Train but skip ONNX export
python -m training --skip-export

# Train + export but skip backend deploy
python -m training --skip-deploy

# Custom hyper-parameters
python -m training --epochs 100 --batch-size 8 --learning-rate 0.001
```

## Dataset Pipeline

All dataset operations are centralized in `data/dataset_utils.py`:

- **Download**: Fetches from Kaggle automatically
- **Convert**: Transforms raw formats (e.g. Supervisely) into canonical structure
- **Validate**: Hard gate — training will not start unless required directories exist

The canonical dataset structure:

```
dataset/
├── classification/
│   ├── caries/
│   └── no_caries/
├── detection/
│   ├── images/
│   ├── labels/
│   └── data.yaml
└── segmentation/
    ├── images/
    └── masks/
```

## Configuration

All tuneable parameters live in `configs/config.py`:

| Parameter             | Default  | Description                     |
|-----------------------|----------|---------------------------------|
| `MOBILENET_IMG_SIZE`  | 224×224  | Input size for MobileNet        |
| `YOLO_IMG_SIZE`       | 640×640  | Input size for YOLO             |
| `UNET_IMG_SIZE`       | 256×256  | Input size for U-Net            |
| `BATCH_SIZE`          | 16       | Training mini-batch size        |
| `EPOCHS`              | 50       | Maximum training epochs         |
| `LEARNING_RATE`       | 1e-4     | Initial learning rate           |
| `SEED`                | 42       | Random seed for reproducibility |

## Design Invariants

- **No dataset bypass**: No training module accepts dataset path arguments
- **Single pipeline**: `ensure_dataset()` is the only dataset entry point
- **Hard validation**: Missing data terminates execution immediately
- **No fallbacks**: Failed dataset acquisition stops the process
