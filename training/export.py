#!/usr/bin/env python3
"""Export trained YOLOv11x models to ONNX FP16.

Usage:
    python export.py --weights work_dirs/yolo11x_1280/weights/best.pt --imgsz 1280
    python export.py --weights work_dirs/yolo11x_1600/weights/best.pt --imgsz 1600
    python export.py --all   # exports all three models
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def export_one(weights, imgsz, device="0"):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, half=True, device=device)

    onnx_path = weights.with_suffix(".onnx")
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"Exported {onnx_path.name} ({size_mb:.1f} MB)")
    return onnx_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--all", action="store_true", help="Export all three models")
    args = parser.parse_args()

    if args.all:
        models = [
            (ROOT / "work_dirs" / "yolo11x_1280" / "weights" / "best.pt", 1280),
            (ROOT / "work_dirs" / "yolo11x_1600" / "weights" / "best.pt", 1600),
            (ROOT / "work_dirs" / "yolo11x_balanced" / "weights" / "best.pt", 1280),
        ]
        for weights, imgsz in models:
            if weights.exists():
                export_one(weights, imgsz, args.device)
            else:
                print(f"Skipping {weights} (not found)")
    else:
        if not args.weights:
            print("Specify --weights or use --all")
            return
        export_one(args.weights, args.imgsz, args.device)


if __name__ == "__main__":
    main()
