#!/usr/bin/env python3
"""Export best.pt to ONNX + classes.txt for CameraYolo."""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    weights = Path(r"D:\OpenCV\develop\1\runs\detect\widerperson_person\weights\best.pt")
    if not weights.exists():
        weights = Path(r"D:\OpenCV\develop\1\runs\detect\widerperson_person\weights\last.pt")
    outdir = Path(r"D:\OpenCV\develop\1\models")
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=640, opset=17, simplify=True))
    target = outdir / "widerperson_person.onnx"
    target.write_bytes(exported.read_bytes())

    names = model.names
    if isinstance(names, dict):
        labels = [names[k] for k in sorted(names)]
    else:
        labels = list(names)
    (outdir / "classes.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    print("ONNX", target, target.stat().st_size)
    print("CLASSES", labels)


if __name__ == "__main__":
    main()
