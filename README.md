# NM i AI 2026 — Task 3: NorgesGruppen Shelf Detection

Score: **0.9174** on the public leaderboard. Final submission tweaks the WBF IoU from 0.65 → 0.60, same models otherwise.

## What we did

We ensemble three YOLOv11x models and merge their predictions with Weighted Box Fusion (WBF). Each model also does a horizontal flip pass (TTA), so there are 6 detection lists per image going into WBF.

| Model | Res | Data | Val mAP |
|-------|-----|------|---------|
| YOLOv11x | 1280 | 90/10 split | 0.757 |
| YOLOv11x | 1600 | 90/10 split | 0.758 |
| YOLOv11x | 1280 | Oversampled (sqrt repeat-factor) | 0.757 |

All exported to ONNX FP16 (~101 MB each → fits 3 under the 420 MB limit).

The inference flow:
1. Run each model at its training resolution
2. Flip TTA doubles the detection lists (6 total)
3. NMS per model (conf > 0.001, IoU 0.5)
4. WBF merges everything (IoU 0.60)
5. Keep top 300 detections, write COCO JSON

We set the confidence threshold very low (0.001) on purpose — WBF handles the scoring, so it's better to feed it more candidates and let fusion suppress the weak ones.

## Files

```
inference/run.py          — the actual submission script (ONNX + WBF + TTA)
training/prepare_data.py  — stratified train/val split
training/oversample.py    — repeat-factor oversampling for rare classes
training/train.py         — trains the three YOLOv11x variants
training/export.py        — .pt → ONNX FP16
training/build_submission.py — zips ONNX models + run.py
```

## How to reproduce

You need Python 3.10+, a GPU with ~32 GB VRAM, and `pip install ultralytics scikit-learn numpy`.

Put the competition data in:
```
data/train/images/          (shelf images)
data/train/labels/          (YOLO-format labels)
data/train/annotations.json (COCO annotations)
```

Then:
```bash
# prep
python training/prepare_data.py --data-dir data
python training/oversample.py --data-dir data

# train (each takes a few hours on an RTX 5090)
python training/train.py --variant standard --device 0
python training/train.py --variant highres  --device 0
python training/train.py --variant balanced --device 0

# export and package
python training/export.py --all --device 0
python training/build_submission.py \
    --models work_dirs/yolo11x_1280/weights/best.onnx \
             work_dirs/yolo11x_1600/weights/best.onnx \
             work_dirs/yolo11x_balanced/weights/best.onnx \
    --output submission.zip
```

## Why these choices

**YOLOv11x** — best accuracy we could get from the YOLO family. We trained with ultralytics 8.4.24 but the sandbox only has 8.1.0, so exporting to ONNX sidesteps that entirely.

**FP16 ONNX** — cuts model size roughly in half (~110 → ~101 MB) which is the only way to squeeze 3 models under the weight limit.

**WBF over custom NMS** — `ensemble_boxes` is pre-installed in the sandbox and gave us +0.006 mAP over a hand-rolled WBF.

**Horizontal flip TTA** — cheap and added ~1.2 mAP in ensemble mode.

**Oversampled third model** — sqrt repeat-factor sampling for rare categories. Doesn't help much on its own but adds diversity to the ensemble.

**1600px model** — barely better as a single model, but the resolution diversity helps the ensemble.

**No second-stage classifier** — we tried detector + ResNet50 classifier but it actually hurt. The 3-YOLO ensemble was already better than 2-YOLO + classifier.

## Sandbox notes

`run.py` only imports stuff that's pre-installed: `cv2`, `numpy`, `onnxruntime`, `ensemble_boxes`, plus stdlib (`argparse`, `json`, `pathlib`). No banned imports.

## Hardware

Training: RTX 5090 32 GB (vast.ai). Inference: L4 24 GB (competition sandbox).
