# NM i AI 2026 — Task 3: Grocery Shelf Product Detection

## Final Score

**0.9174** public leaderboard (v4 baseline with WBF IoU 0.65).  
Final submission: v5 variant with WBF IoU 0.60 (same models, minor WBF tuning).

## Approach

**3× YOLOv11x ensemble** with Weighted Box Fusion (WBF) and horizontal flip TTA.

| Model | Resolution | Training Data | Val mAP@0.5 |
|-------|-----------|---------------|-------------|
| YOLOv11x | 1280px | Standard 90/10 split | 0.757 |
| YOLOv11x | 1600px | Standard 90/10 split | 0.758 |
| YOLOv11x | 1280px | Class-balanced (sqrt oversampling) | 0.757 |

All models exported to ONNX FP16 (~101 MB each, 3 models fit under 420 MB limit).

### Ensemble Pipeline

1. Each model runs inference at its native resolution
2. Horizontal flip TTA doubles detections per model (6 detection lists total)
3. Per-model NMS filters candidates (conf > 0.001, IoU 0.5, top 900)
4. `ensemble_boxes` WBF merges all 6 lists (IoU threshold 0.60)
5. Top 300 detections per image output as COCO JSON

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| CONF_THRESH | 0.001 | Low to feed more candidates to WBF |
| IOU_THRESH | 0.5 | Per-model NMS |
| WBF_IOU_THRESH | 0.60 | WBF merging threshold |
| MAX_DET | 300 | Final detections per image |
| ENABLE_TTA | True | Horizontal flip |

## Project Structure

```
delivery_final/
├── README.md               # This file
├── inference/
│   └── run.py              # Submission inference code (ONNX + WBF + TTA)
└── training/
    ├── prepare_data.py      # Stratified train/val split
    ├── oversample.py        # Class-balanced repeat-factor sampling
    ├── train.py             # Train 3× YOLOv11x variants
    ├── export.py            # Export .pt → ONNX FP16
    └── build_submission.py  # Package ONNX models + run.py into zip
```

## Reproducing Results

### Requirements

- Python 3.10+
- GPU: NVIDIA RTX 5090 32GB (or equivalent, ~32GB VRAM)
- `pip install ultralytics scikit-learn numpy`

### 1. Data Setup

```
data/train/images/           — shelf images (img_XXXXX.jpg)
data/train/labels/           — YOLO format labels (img_XXXXX.txt)
data/train/annotations.json  — COCO annotations
```

### 2. Prepare Data

```bash
python training/prepare_data.py --data-dir data
python training/oversample.py --data-dir data
```

### 3. Train Models

```bash
python training/train.py --variant standard --device 0   # ~3h on RTX 5090
python training/train.py --variant highres  --device 0   # ~5h on RTX 5090
python training/train.py --variant balanced --device 0   # ~3h on RTX 5090
```

### 4. Export & Build Submission

```bash
python training/export.py --all --device 0

python training/build_submission.py \
    --models work_dirs/yolo11x_1280/weights/best.onnx \
             work_dirs/yolo11x_1600/weights/best.onnx \
             work_dirs/yolo11x_balanced/weights/best.onnx \
    --output submission.zip
```

## Key Design Decisions

- **YOLOv11x**: Best accuracy in the YOLO family. Exported to ONNX to bypass sandbox ultralytics version constraint (sandbox has 8.1.0, we trained with 8.4.24).
- **FP16 ONNX**: Halves model size (~110 MB → ~101 MB each) to fit 3 models under the 420 MB submission limit.
- **ensemble_boxes WBF**: Pre-installed in sandbox (v1.0.9). Produces better merged boxes than custom WBF (+0.006 on val).
- **Very low CONF_THRESH (0.001)**: WBF handles scoring — low-confidence candidates get suppressed by the fusion process.
- **Horizontal flip TTA**: +1.2 mAP in ensemble mode. Cheap at inference time.
- **Class-balanced oversampling**: Repeat-factor sampling with sqrt scaling for rare categories. Adds diversity to ensemble.
- **1600px model**: Marginal single-model gain but adds resolution diversity.
- **No classifier stage**: Tested 2-stage (detector + ResNet50 classifier) but it hurt the score — the 3-YOLO ensemble already outperformed 2-YOLO + classifier.

## Sandbox Compliance

Submission `run.py` uses only sandbox-allowed imports:
- `argparse`, `json`, `pathlib` (stdlib)
- `cv2`, `numpy`, `onnxruntime` (pre-installed)
- `ensemble_boxes` (pre-installed v1.0.9)

No banned imports (`os`, `sys`, `subprocess`, etc.).

## Hardware

- **Training**: NVIDIA RTX 5090 32GB (vast.ai)
- **Inference**: NVIDIA L4 24GB (competition sandbox, 300s timeout)
