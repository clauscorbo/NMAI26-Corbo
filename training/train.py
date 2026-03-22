#!/usr/bin/env python3
"""Train YOLOv11x models for shelf product detection.

Three variants:
  1. Standard   — 1280px, standard data
  2. High-res   — 1600px, standard data
  3. Balanced   — 1280px, class-balanced oversampled data

Usage:
    python train.py --variant standard --device 0
    python train.py --variant highres  --device 0
    python train.py --variant balanced --device 0
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
WORK_DIRS = ROOT / "work_dirs"

VARIANTS = {
    "standard": {
        "data": CONFIGS / "shelf_yolo_data.yaml",
        "imgsz": 1280,
        "name": "yolo11x_1280",
    },
    "highres": {
        "data": CONFIGS / "shelf_yolo_data.yaml",
        "imgsz": 1600,
        "name": "yolo11x_1600",
    },
    "balanced": {
        "data": CONFIGS / "shelf_yolo_balanced.yaml",
        "imgsz": 1280,
        "name": "yolo11x_balanced",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS.keys(), required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--model", default="yolo11x.pt")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = VARIANTS[args.variant]

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(cfg["data"]),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=cfg["imgsz"],
        device=args.device,
        project=str(WORK_DIRS),
        name=cfg["name"],
        exist_ok=True,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        erasing=0.4,
        close_mosaic=10,
        # Loss weights
        box=7.5,
        cls=1.5,
        dfl=1.5,
        # Detection
        max_det=500,
        # Other
        amp=True,
        patience=30,
        save_period=10,
        resume=args.resume,
        workers=8,
    )


if __name__ == "__main__":
    main()
