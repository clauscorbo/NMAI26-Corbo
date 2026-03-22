#!/usr/bin/env python3
"""Prepare COCO dataset into stratified train/val splits with YOLO-format labels.

Usage:
    python prepare_data.py --data-dir ../data

Expects:
    data/train/images/       — shelf images (img_XXXXX.jpg)
    data/train/labels/       — YOLO labels (img_XXXXX.txt)
    data/train/annotations.json — COCO annotations

Produces:
    data/splits/train/images/  data/splits/train/labels/
    data/splits/val/images/    data/splits/val/labels/
    configs/shelf_yolo_data.yaml
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def load_annotations(data_dir):
    with open(data_dir / "train" / "annotations.json") as f:
        return json.load(f)


def create_stratified_split(ann, n_splits=5, val_fold=0, seed=42):
    """90/10 stratified split using dominant category per image."""
    img_cats = defaultdict(list)
    for a in ann["annotations"]:
        img_cats[a["image_id"]].append(a["category_id"])

    image_ids, dominant_cats = [], []
    for img in ann["images"]:
        if img["id"] in img_cats:
            dominant = Counter(img_cats[img["id"]]).most_common(1)[0][0]
            image_ids.append(img["id"])
            dominant_cats.append(dominant)

    image_ids = np.array(image_ids)
    dominant_cats = np.array(dominant_cats)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = list(splitter.split(image_ids, dominant_cats, groups=image_ids))[val_fold]

    return set(image_ids[train_idx].tolist()), set(image_ids[val_idx].tolist())


def symlink_split(ann, image_ids, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    """Symlink images and labels for a given split."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    id_to_file = {img["id"]: img["file_name"] for img in ann["images"]}
    count = 0
    for img_id in image_ids:
        fname = id_to_file.get(img_id)
        if not fname:
            continue
        stem = Path(fname).stem

        src_img = src_img_dir / fname
        src_lbl = src_lbl_dir / f"{stem}.txt"

        if src_img.exists():
            dst = dst_img_dir / fname
            if not dst.exists():
                dst.symlink_to(src_img.resolve())
            count += 1

        if src_lbl.exists():
            dst = dst_lbl_dir / f"{stem}.txt"
            if not dst.exists():
                dst.symlink_to(src_lbl.resolve())

    return count


def write_yolo_yaml(ann, splits_dir, yaml_path):
    """Write YOLO dataset YAML config."""
    cat_map = {c["id"]: c["name"] for c in ann["categories"]}
    lines = [
        f"path: {splits_dir}",
        "train: train/images",
        "val: val/images",
        "",
        f"nc: {len(cat_map)}",
        "",
        "names:",
    ]
    for i in range(len(cat_map)):
        name = cat_map.get(i, f"class_{i}").replace("'", "").replace('"', "").replace(":", " -")
        lines.append(f'  {i}: "{name}"')

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = parser.parse_args()

    data_dir = args.data_dir
    ann = load_annotations(data_dir)
    print(f"Loaded {len(ann['images'])} images, {len(ann['annotations'])} annotations, {len(ann['categories'])} categories")

    train_ids, val_ids = create_stratified_split(ann)
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val")

    splits_dir = data_dir / "splits"
    src_img = data_dir / "train" / "images"
    src_lbl = data_dir / "train" / "labels"

    n_train = symlink_split(ann, train_ids, src_img, src_lbl,
                            splits_dir / "train" / "images", splits_dir / "train" / "labels")
    n_val = symlink_split(ann, val_ids, src_img, src_lbl,
                          splits_dir / "val" / "images", splits_dir / "val" / "labels")
    print(f"Linked {n_train} train, {n_val} val images")

    yaml_path = Path(__file__).resolve().parent.parent / "configs" / "shelf_yolo_data.yaml"
    write_yolo_yaml(ann, splits_dir, yaml_path)
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
