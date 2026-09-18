#!/usr/bin/env python3
"""Convert WiderPerson COCO-json package to Ultralytics YOLO layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_split(images_dir: Path, coco_json: Path, labels_dir: Path) -> int:
    with coco_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = {im["id"]: im for im in data.get("images", [])}
    anns_by_image: dict[int, list] = {iid: [] for iid in images}
    for ann in data.get("annotations", []):
        iid = ann.get("image_id")
        if iid in anns_by_image and not ann.get("iscrowd", 0):
            anns_by_image[iid].append(ann)

    labels_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for iid, im in images.items():
        w = float(im["width"])
        h = float(im["height"])
        if w <= 0 or h <= 0:
            continue
        lines = []
        for ann in anns_by_image[iid]:
            x, y, bw, bh = ann["bbox"]
            # skip degenerate boxes
            if bw <= 1 or bh <= 1:
                continue
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            # clamp
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines:
            continue

        stem = Path(im["file_name"]).stem
        # ensure image exists
        src = images_dir / Path(im["file_name"]).name
        if not src.exists():
            # try as-is relative
            src = images_dir / im["file_name"]
        if not src.exists():
            continue

        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=r"D:\OpenCV\develop\1\data\widerperson_coco")
    p.add_argument("--out", default=r"D:\OpenCV\develop\1\data\widerperson_yolo")
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    splits = {
        "train": (root / "train", root / "annotations" / "train.json"),
        "val": (root / "val", root / "annotations" / "valid.json"),
    }

    for name, (img_dir, ann) in splits.items():
        if not ann.exists():
            raise SystemExit(f"missing annotation: {ann}")
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)

        # symlink/copy images is slow on Windows; use hardlink when possible
        imgs = list(img_dir.glob("*.jpg"))
        linked = 0
        for img in imgs:
            dst = out / "images" / name / img.name
            if dst.exists():
                linked += 1
                continue
            try:
                dst.hardlink_to(img)
            except OSError:
                import shutil

                shutil.copy2(img, dst)
            linked += 1

        n = convert_split(img_dir, ann, out / "labels" / name)
        print(f"[{name}] images={linked} labeled={n}")

    yaml = f"""path: {out.as_posix()}
train: images/train
val: images/val
names:
  0: person
"""
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"[OK] wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
