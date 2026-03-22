#!/usr/bin/env python3
"""Build submission zip from ONNX models + run.py.

Usage:
    python build_submission.py \
        --models model_1280.onnx model_1600.onnx model_bal.onnx \
        --output submission.zip
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=Path, required=True,
                        help="ONNX model files to include")
    parser.add_argument("--run-py", type=Path,
                        default=ROOT / "inference" / "run.py")
    parser.add_argument("--output", type=Path, default=Path("submission.zip"))
    args = parser.parse_args()

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(args.run_py, "run.py")
        for model in args.models:
            zf.write(model, f"model_weights/{model.name}")

    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"Built {args.output} ({size_mb:.1f} MB)")

    # Verify structure
    with zipfile.ZipFile(args.output) as zf:
        names = zf.namelist()
        assert "run.py" in names, "run.py missing from zip root"
        onnx_count = sum(1 for n in names if n.endswith(".onnx"))
        print(f"  Files: {len(names)}, ONNX models: {onnx_count}")
        assert onnx_count <= 3, f"Too many ONNX models ({onnx_count} > 3)"


if __name__ == "__main__":
    main()
