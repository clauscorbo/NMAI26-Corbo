# NM i AI 2026 — Task 3

3x YOLOv11x ensemble with WBF and horizontal flip TTA. Score: **0.9177**.

## Approach

| Model | Res | Data | Val mAP |
|-------|-----|------|---------|
| YOLOv11x | 1280 | 90/10 split | 0.757 |
| YOLOv11x | 1600 | 90/10 split | 0.758 |
| YOLOv11x | 1280 | Oversampled (sqrt repeat-factor) | 0.757 |

All exported to ONNX FP16 (~101 MB each). Trained with ultralytics 8.4.24, exported to ONNX to avoid the sandbox version mismatch.

Each model runs at its training resolution, flip TTA doubles the detection lists (6 total), then WBF merges everything (IoU 0.60). Confidence threshold is very low (0.001) -- WBF handles the scoring.

We tried a 2-stage approach (detector + ResNet50 classifier) but the 3-model ensemble already beat it.

## Files

- `inference/run.py` -- submission script
- `training/prepare_data.py` -- stratified train/val split
- `training/oversample.py` -- repeat-factor oversampling for rare classes
- `training/train.py` -- trains the three variants
- `training/export.py` -- .pt to ONNX FP16
- `training/build_submission.py` -- zips models + run.py

## Reproduce

Needs Python 3.10+, ~32 GB VRAM GPU, `pip install ultralytics scikit-learn numpy`.

```bash
python training/prepare_data.py --data-dir data
python training/oversample.py --data-dir data

python training/train.py --variant standard --device 0
python training/train.py --variant highres  --device 0
python training/train.py --variant balanced --device 0

python training/export.py --all --device 0
python training/build_submission.py \
    --models work_dirs/yolo11x_1280/weights/best.onnx \
             work_dirs/yolo11x_1600/weights/best.onnx \
             work_dirs/yolo11x_balanced/weights/best.onnx \
    --output submission.zip
```

Trained on RTX 5090 32 GB (vast.ai).
