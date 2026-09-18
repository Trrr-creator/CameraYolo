#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容入口：CPU 子集调试脚本。

正式训练请使用:
  training/train_yolov8.py
（自动定位 conda「pytorch」环境 CUDA / 低学习率 / 高轮数 / 中文可视化）

本脚本保留用于快速子集调试，默认仅打印用法，不自动开训。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="调试入口：转发到 train_yolov8.py（默认不自动训练）",
    )
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--name", default="widerperson_person_debug")
    parser.add_argument("--data", default=r"D:\OpenCV\develop\1\data\widerperson_yolo\data.yaml")
    parser.add_argument("--model", default=r"D:\OpenCV\develop\1\models\yolov8n.pt")
    parser.add_argument("--run", action="store_true", help="真正执行训练；默认只打印命令")
    args = parser.parse_args()

    script = Path(__file__).resolve().parent / "train_yolov8.py"
    cmd = [
        sys.executable,
        str(script),
        "--data",
        args.data,
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--name",
        args.name,
        "--subset",
        str(args.max_train),
        "--lr0",
        "0.001",
        "--patience",
        "20",
        "--force-pytorch-env",
    ]
    if args.device:
        cmd += ["--device", args.device]
    else:
        cmd += ["--allow-cpu"]

    print("=" * 64)
    print("【训练入口】正式脚本位置:")
    print(f"  {script}")
    print("【当前解释器】")
    print(f"  {sys.executable}")
    print("【conda CUDA 环境】请优先使用:")
    print(r"  D:\Anaconda\envs\pytorch\python.exe")
    print("【建议命令】")
    pretty = " ".join(cmd)
    print(f"  {pretty}")
    print("=" * 64)
    print("说明：本脚本默认【不会】自动训练。")
    print("      确认无误后，将上面命令贴到终端执行，或运行时加 --run。")

    if not args.run:
        return

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
