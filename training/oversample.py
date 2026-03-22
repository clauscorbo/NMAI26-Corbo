#!/usr/bin/env python3
"""Create class-balanced oversampled training set using repeat-factor sampling.

Rare classes are oversampled by duplicating images that contain them.
Repeat factor per image = max over its categories of sqrt(target / count).

Usage:
    python oversample.py --data-dir ../data
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = parser.parse_args()

    splits_dir = args.data_dir / "splits"
    train_json = splits_dir / "train.json" if (splits_dir / "train.json").exists() else None

    # Build category counts from label files
    train_lbl_dir = splits_dir / "train" / "labels"
    train_img_dir = splits_dir / "train" / "images"

    img_cats = {}
    cat_counts = Counter()
    for lbl_file in sorted(train_lbl_dir.glob("*.txt")):
        cats = set()
        for line in lbl_file.read_text().strip().split("\n"):
            if line.strip():
                cats.add(int(line.split()[0]))
        img_cats[lbl_file.stem] = cats
        cat_counts.update(cats)

    median = sorted(cat_counts.values())[len(cat_counts) // 2]
    target_count = max(20, median)
    print(f"Categories: {len(cat_counts)}, median count: {median}, target: {target_count}")

    # Calculate repeat factor per image
    img_repeat = {}
    for stem, cats in img_cats.items():
        max_factor = 1.0
        for cat in cats:
            count = cat_counts[cat]
            if count < target_count:
                factor = (target_count / count) ** 0.5
                max_factor = max(max_factor, factor)
        img_repeat[stem] = max(1, int(round(max_factor)))

    oversampled = sum(1 for v in img_repeat.values() if v > 1)
    total = sum(img_repeat.values())
    print(f"Oversampled: {oversampled}/{len(img_repeat)} images, total: {total}")

    # Create oversampled directory with symlinks
    os_img = splits_dir / "train_balanced" / "images"
    os_lbl = splits_dir / "train_balanced" / "labels"
    os_img.mkdir(parents=True, exist_ok=True)
    os_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for stem, repeats in img_repeat.items():
        # Find source image (jpg or jpeg)
        src_img = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = train_img_dir / f"{stem}{ext}"
            if candidate.exists():
                src_img = candidate
                break
        if not src_img:
            continue

        src_lbl = train_lbl_dir / f"{stem}.txt"

        for rep in range(repeats):
            suffix = "" if rep == 0 else f"_r{rep}"
            dst_img = os_img / f"{stem}{suffix}{src_img.suffix}"
            dst_lbl = os_lbl / f"{stem}{suffix}.txt"

            if not dst_img.exists():
                dst_img.symlink_to(src_img.resolve())
            if src_lbl.exists() and not dst_lbl.exists():
                dst_lbl.symlink_to(src_lbl.resolve())
            count += 1

    print(f"Created {count} image+label pairs in {os_img}")

    # Write YOLO data config for balanced training
    with open(args.data_dir / "train" / "annotations.json") as f:
        ann = json.load(f)
    cat_map = {c["id"]: c["name"] for c in ann["categories"]}

    lines = [
        f"path: {splits_dir}",
        "train: train_balanced/images",
        "val: val/images",
        "",
        f"nc: {len(cat_map)}",
        "",
        "names:",
    ]
    for i in range(len(cat_map)):
        name = cat_map.get(i, f"class_{i}").replace("'", "").replace('"', "").replace(":", " -")
        lines.append(f'  {i}: "{name}"')

    yaml_path = Path(__file__).resolve().parent.parent / "configs" / "shelf_yolo_balanced.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
