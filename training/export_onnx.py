#!/usr/bin/env python3
"""Export a trained Ultralytics model to ONNX + classes.txt for CameraYolo."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO weights to ONNX for CameraYolo")
    p.add_argument("--weights", required=True, help="path to best.pt")
    p.add_argument("--name", default="custom_yolo", help="output base name")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--outdir", default="models")
    return p.parse_args()


def write_classes_from_model(model, out_path: Path) -> int:
    names = getattr(model, "names", None)
    if names is None:
        return 0
    # names may be dict {id: name} or list
    if isinstance(names, dict):
        labels = [names[k] for k in sorted(names.keys())]
    else:
        labels = list(names)
    out_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return len(labels)


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("请先: pip install ultralytics") from exc

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"找不到权重: {weights}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    export_path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset)
    export_path = Path(export_path)

    target_onnx = outdir / f"{args.name}.onnx"
    shutil.copy2(export_path, target_onnx)

    classes = outdir / "classes.txt"
    n = write_classes_from_model(model, classes)

    print(f"[OK] ONNX  : {target_onnx.resolve()}")
    print(f"[OK] 类别数: {n} -> {classes.resolve()}")
    print("在 CameraYolo 中点击「加载模型…」选择上述 onnx 即可。")


if __name__ == "__main__":
    main()
