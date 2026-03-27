# Dental Caries Detection – Offline Training Pipeline

This directory contains the **offline** training pipeline for training and
evaluating deep-learning models that detect early-stage dental caries from
X-ray images.  It is completely separate from the web application and is
intended to be run on a machine with a GPU.

## Directory Structure

```
training/
├── configs/
│   └── config.py              # All hyper-parameters, paths, and constants
├── data/
│   ├── dataset.py             # Dataset discovery, loading, and splitting
│   └── augmentation.py        # X-ray-specific data augmentations
├── preprocessing/
│   └── preprocess.py          # CLAHE enhancement, resize, normalise
├── training/
│   ├── train_mobilenet.py     # MobileNetV2 classification (TensorFlow)
│   ├── train_yolo.py          # YOLOv8 detection (Ultralytics)
│   └── train_unet.py          # U-Net segmentation (TensorFlow)
├── evaluation/
│   └── evaluate.py            # Metrics: accuracy, precision, recall, F1, IoU
├── models/                    # Saved model checkpoints (git-ignored)
├── outputs/                   # TensorBoard logs and run artefacts
├── metrics/                   # Evaluation JSON reports
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

### 1. Install dependencies

```bash
cd training/
pip install -r requirements.txt
```

### 2. Prepare your dataset

For **classification** (MobileNet), organise images as:

```
dataset/
├── No Caries/
│   ├── img_001.png
│   └── ...
└── Caries/
    ├── img_001.png
    └── ...
```

For **detection** (YOLO), use the standard YOLO format with a `data.yaml`.

For **segmentation** (U-Net), provide paired image and mask directories
where files share the same stem (e.g. `img_001.png` in both dirs).

### 3. Train a model

**MobileNet classification:**

```bash
python -m training.training.train_mobilenet \
    --data-dir /path/to/classification/dataset \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 0.0001
```

**YOLO detection:**

```bash
python -m training.training.train_yolo \
    --data /path/to/data.yaml \
    --epochs 100 \
    --batch-size 16
```

**U-Net segmentation:**

```bash
python -m training.training.train_unet \
    --image-dir /path/to/images \
    --mask-dir /path/to/masks \
    --epochs 50 \
    --batch-size 8
```

### 4. Evaluate a model

```bash
# MobileNet
python -m training.evaluation.evaluate \
    --model-path models/mobilenet_best.keras \
    --data-dir /path/to/dataset \
    --model-type mobilenet

# U-Net
python -m training.evaluation.evaluate \
    --model-path models/unet_best.keras \
    --data-dir /path/to/images \
    --mask-dir /path/to/masks \
    --model-type unet
```

Evaluation results are saved as JSON in the `metrics/` directory.

## Configuration

All tuneable parameters live in `configs/config.py`:

| Parameter               | Default  | Description                         |
|------------------------|----------|-------------------------------------|
| `MOBILENET_IMG_SIZE`   | 224×224  | Input size for MobileNet            |
| `YOLO_IMG_SIZE`        | 640×640  | Input size for YOLO                 |
| `UNET_IMG_SIZE`        | 256×256  | Input size for U-Net                |
| `BATCH_SIZE`           | 16       | Training mini-batch size            |
| `EPOCHS`               | 50       | Maximum training epochs             |
| `LEARNING_RATE`        | 1e-4     | Initial learning rate               |
| `TRAIN_RATIO`          | 0.70     | Training split ratio                |
| `VAL_RATIO`            | 0.15     | Validation split ratio              |
| `TEST_RATIO`           | 0.15     | Test split ratio                    |
| `SEED`                 | 42       | Random seed for reproducibility     |

## Models

- **MobileNetV2** – lightweight classification (transfer learning from ImageNet)
- **YOLOv8** – real-time object detection (via Ultralytics)
- **U-Net** – pixel-level segmentation of carious regions
