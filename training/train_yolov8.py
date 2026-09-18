#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 自定义数据集训练脚本（推荐入口）

特性
----
1. 自动定位 conda 环境 “pytorch” 下的 CUDA 版 PyTorch（若当前解释器不是则自动切换重跑）
2. 默认降低学习率、提高训练轮数
3. 优化早停：基于 mAP50-95 + 最小提升量 min_delta + 冷却期
4. 终端打印训练进度与中文指标表
5. 训练结束后用 matplotlib 绘制中文可视化：
   损失曲线、Precision/Recall、mAP、混淆矩阵热力图
6. Windows 控制台/文件统一 UTF-8，降低中文乱码风险

"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 编码：Windows 终端 / matplotlib 中文兼容
# ---------------------------------------------------------------------------
def setup_utf8_console() -> None:
    if sys.platform.startswith("win"):
        os.system("")  # 启用 ANSI（Win10+）
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            if stream is None:
                continue
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")


def setup_openmp_workaround() -> None:
    """
    Anaconda/torch 常见问题：
      OMP: Error #15 Initializing libiomp5md.dll ... already initialized
      → DataLoader worker 崩溃 (worker exited unexpectedly)

    策略：
      1. 允许重复 OpenMP 运行时（Intel 建议的 unsafe workaround，对训练通常可用）
      2. 降低默认 worker，避免多进程再叠一层 OpenMP 冲突
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    # 压制 Ultralytics / torch 冗长日志
    os.environ.setdefault("YOLO_VERBOSE", "false")
    os.environ.setdefault("ULTRALYTICS_OFFLINE", "0")


# ---------------------------------------------------------------------------
# 自动定位 conda「pytorch」环境
# ---------------------------------------------------------------------------
def find_pytorch_env_python() -> Path | None:
    """搜索本机常见 Anaconda/Miniconda 路径下的 envs/pytorch/python.exe。"""
    candidates: list[Path] = []

    # 显式环境变量优先
    for key in ("PYTORCH_ENV_PYTHON", "CONDA_PYTORCH_PYTHON"):
        val = os.environ.get(key)
        if val:
            candidates.append(Path(val))

    home = Path.home()
    userprofile = Path(os.environ.get("USERPROFILE", home))
    drives = ["C:", "D:", "E:"]

    patterns = []
    for d in drives:
        patterns += [
            Path(f"{d}\\Anaconda3\\envs\\pytorch\\python.exe"),
            Path(f"{d}\\Anaconda\\envs\\pytorch\\python.exe"),
            Path(f"{d}\\Miniconda3\\envs\\pytorch\\python.exe"),
            Path(f"{d}\\miniconda3\\envs\\pytorch\\python.exe"),
            Path(f"{d}\\ProgramData\\Anaconda3\\envs\\pytorch\\python.exe"),
        ]
    patterns += [
        home / "anaconda3" / "envs" / "pytorch" / "python.exe",
        home / "Anaconda3" / "envs" / "pytorch" / "python.exe",
        home / "miniconda3" / "envs" / "pytorch" / "python.exe",
        userprofile / "anaconda3" / "envs" / "pytorch" / "python.exe",
        userprofile / "Anaconda3" / "envs" / "pytorch" / "python.exe",
    ]

    # 若当前脚本在某个 env 内，尝试兄弟环境
    try:
        exe = Path(sys.executable)
        # .../envs/xxx/python.exe
        if exe.parent.parent.name.lower() == "envs":
            candidates.append(exe.parent.parent / "pytorch" / "python.exe")
    except Exception:
        pass

    candidates.extend(patterns)
    for p in candidates:
        try:
            if p.is_file():
                return p.resolve()
        except Exception:
            continue
    return None


def probe_interpreter(python: Path) -> dict:
    """探测某解释器的 torch/CUDA，并做一次真实 GPU 算子冒烟测试。"""
    code = r"""
import json, sys
info = {"exe": sys.executable, "ok": False, "torch": None, "cuda": False,
        "gpu": None, "capability": None, "cuda_smoke": False, "error": None}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda"] = bool(torch.cuda.is_available())
    if info["cuda"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["capability"] = f"sm_{cap[0]}{cap[1]}"
        # 真实算子：避免 only-is_available=true 但无 kernel（如 sm_120 + 旧 torch）
        try:
            x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
            y = (x @ x).sum()
            torch.cuda.synchronize()
            info["cuda_smoke"] = bool(y.numel() == 1)
            info["smoke_value"] = float(y.detach().cpu())
        except Exception as se:
            info["cuda_smoke"] = False
            info["error"] = f"GPU算子失败: {se}"
except Exception as e:
    info["error"] = str(e)
print("__JSON__" + json.dumps(info, ensure_ascii=False))
"""
    try:
        proc = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        out = proc.stdout or ""
        if "__JSON__" in out:
            import json

            return json.loads(out.split("__JSON__", 1)[1].strip().splitlines()[0])
        return {"exe": str(python), "ok": False, "error": (proc.stderr or out)[-400:]}
    except Exception as e:
        return {"exe": str(python), "ok": False, "error": str(e)}


def print_sm120_fix_guide(info: dict | None = None) -> None:
    print("-" * 64)
    print("【GPU 不可用 / kernel 缺失修复指南】")
    print("现象：torch.cuda.is_available()=True，但训练报")
    print("      RuntimeError: CUDA error: no kernel image is available for execution on the device")
    print("原因：当前 PyTorch 的 CUDA wheel 不包含该 GPU 架构的 kernel（RTX 50 系 = sm_120）。")
    if info:
        print(f"  当前 torch: {info.get('torch')}")
        print(f"  GPU: {info.get('gpu')}  {info.get('capability')}")
        if info.get("error"):
            print(f"  探测错误: {info['error']}")
    print("修复（在 conda pytorch 环境中执行）：")
    print(r"  D:\Anaconda\envs\pytorch\python.exe -m pip uninstall -y torch torchvision")
    print(r"  D:\Anaconda\envs\pytorch\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    print("国内网络可试：")
    print("  https://mirrors.aliyun.com/pytorch-wheels/cu128/")
    print("  下载 torch-2.10.0+cu128-cp311-cp311-win_amd64.whl 后:")
    print(r"  D:\Anaconda\envs\pytorch\python.exe -m pip install <whl路径>")
    print("临时方案：加 --allow-cpu 用 CPU 训练（慢）。")
    print("-" * 64)


def ensure_cuda_pytorch(allow_cpu: bool, force_pytorch_env: bool) -> Path:
    """定位/切换到 conda「pytorch」环境的可用 CUDA。输出仅保留设备摘要。"""
    current = Path(sys.executable).resolve()
    current_info = probe_interpreter(current)
    cuda_ok = bool(current_info.get("cuda") and current_info.get("cuda_smoke"))

    def brief(info: dict, tag: str) -> str:
        gpu = info.get("gpu") or "CPU"
        cap = info.get("capability") or "-"
        torch_v = info.get("torch") or "-"
        smoke = "OK" if info.get("cuda_smoke") else "FAIL"
        return f"设备[{tag}] {gpu} {cap} | torch {torch_v} | CUDA冒烟 {smoke}"

    print(brief(current_info, "当前"))

    if cuda_ok and not force_pytorch_env:
        return current

    pytorch_py = find_pytorch_env_python()
    if pytorch_py is None:
        if allow_cpu:
            print("未找到 conda pytorch，回退 CPU")
            return current
        print_sm120_fix_guide(current_info)
        raise SystemExit("未找到 conda 环境 pytorch 的 python.exe")

    if pytorch_py.resolve() == current:
        if cuda_ok:
            return current
        if allow_cpu:
            print("pytorch 环境 CUDA 不可用，使用 CPU")
            print_sm120_fix_guide(current_info)
            return current
        print_sm120_fix_guide(current_info)
        raise SystemExit("conda pytorch 的 CUDA 不可用或缺少 GPU kernel")

    info = probe_interpreter(pytorch_py)
    print(brief(info, "pytorch"))

    if not (info.get("cuda") and info.get("cuda_smoke")):
        if allow_cpu:
            print("pytorch CUDA 冒烟未通过，使用 CPU")
            print_sm120_fix_guide(info)
            return pytorch_py
        print_sm120_fix_guide(info)
        raise SystemExit("conda pytorch CUDA 冒烟失败，请升级 torch (cu128) 或加 --allow-cpu")

    script = Path(__file__).resolve()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["TRAIN_SKIP_ENV_SWITCH"] = "1"
    cmd = [str(pytorch_py), str(script), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, env=env))


# ---------------------------------------------------------------------------
# 早停：在 Ultralytics patience 之上叠加 min_delta 监控
# ---------------------------------------------------------------------------
class EarlyStopMonitor:
    """记录最优 mAP50-95；连续 patience 轮提升 < min_delta 则请求停止。"""

    def __init__(self, patience: int = 20, min_delta: float = 0.002, monitor: str = "metrics/mAP50-95(B)"):
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.best = -1.0
        self.best_epoch = -1
        self.bad_rounds = 0
        self.history: list[dict] = []

    def update(self, row: dict) -> bool:
        self.history.append(row)
        try:
            value = float(row.get(self.monitor, row.get("metrics/mAP50-95(B)", -1.0)))
        except Exception:
            value = -1.0
        if value > self.best + self.min_delta:
            self.best = value
            self.best_epoch = int(row.get("epoch", len(self.history)))
            self.bad_rounds = 0
            return False
        self.bad_rounds += 1
        return self.bad_rounds >= self.patience


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO 训练：自动定位 pytorch 环境 CUDA / 低学习率 / 高轮数 / 中文可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default=r"D:\OpenCV\develop\1\data\widerperson_yolo\data.yaml",
                   help="Ultralytics data.yaml 路径")
    p.add_argument("--model", default=r"D:\OpenCV\develop\1\models\yolov8n.pt",
                   help="基础权重 yolov8n.pt / yolov8s.pt 等")
    p.add_argument("--epochs", type=int, default=100, help="训练轮数（已提高默认值）")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="widerperson_person_gpu")
    p.add_argument("--project", default=r"D:\OpenCV\develop\1\runs\detect")
    p.add_argument("--device", default="", help="如 0 / 0,1 / cpu；空=自动 CUDA0")
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader 进程数；默认 0，避免 OpenMP 多副本导致 worker 崩溃")
    p.add_argument("--lr0", type=float, default=0.001, help="初始学习率（已降低默认值）")
    p.add_argument("--lrf", type=float, default=0.01, help="最终学习率比例 lr0*lrf")
    p.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam", "AdamW"],
                   help="优化器（默认 SGD，避免 auto 忽略 lr0）")
    p.add_argument("--amp", action="store_true",
                   help="开启自动混合精度；默认关闭，避免离线时去 GitHub 下载 AMP 检查权重")
    p.add_argument("--patience", type=int, default=25, help="自定义早停耐心（轮）")
    p.add_argument("--min-delta", type=float, default=0.002, help="mAP50-95 最小提升量")
    p.add_argument("--ultralytics-patience", type=int, default=50,
                   help="传给 Ultralytics 的 patience（放宽，真正早停交给本脚本）")
    p.add_argument("--subset", type=int, default=0, help=">0 时仅使用前 N 张训练图（调试）")
    p.add_argument("--allow-cpu", action="store_true", help="允许在无 CUDA 时 CPU 训练")
    p.add_argument("--force-pytorch-env", action="store_true", help="强制使用 conda pytorch 环境")
    p.add_argument("--no-vis", action="store_true", help="跳过训练后 matplotlib 可视化")
    p.add_argument("--save-dir-hint", default="", help="仅用于可视化已有目录时指定 runs 目录")
    return p.parse_args()


def apply_subset(data_yaml: Path, subset_n: int) -> Path:
    """创建只含前 N 张图的 data.yaml（不改动原数据集）。"""
    if subset_n <= 0:
        return data_yaml
    import yaml

    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data["path"])
    train_rel = data.get("train", "images/train")
    val_rel = data.get("val", "images/val")
    train_dir = root / train_rel
    if not train_dir.is_dir():
        raise SystemExit(f"训练图目录不存在: {train_dir}")

    images = sorted(train_dir.glob("*.jpg"))[:subset_n]
    if not images:
        raise SystemExit(f"子集为空: {train_dir}")

    out_dir = root / f"_subset_{subset_n}"
    img_out = out_dir / "images" / "train"
    lbl_out = out_dir / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    train_label_dir = Path(str(train_dir).replace("images", "labels"))
    for img in images:
        dst = img_out / img.name
        if not dst.exists():
            try:
                dst.hardlink_to(img)
            except OSError:
                import shutil

                shutil.copy2(img, dst)
        lbl = train_label_dir / f"{img.stem}.txt"
        dl = lbl_out / f"{img.stem}.txt"
        if lbl.exists() and not dl.exists():
            try:
                dl.hardlink_to(lbl)
            except OSError:
                import shutil

                shutil.copy2(lbl, dl)

    names = data.get("names") or {0: "person"}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    # path 用原 root，train 指向子集相对路径，val 保持原值
    train_rel_new = (img_out.relative_to(root)).as_posix()
    lines = [
        f"path: {root.as_posix()}",
        f"train: {train_rel_new}",
        f"val: {val_rel}",
        "names:",
    ]
    for k in sorted(names, key=lambda x: int(x)):
        lines.append(f"  {k}: {names[k]}")

    out_yaml = out_dir / "data.yaml"
    out_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[子集] train={len(images)} yaml={out_yaml}")
    return out_yaml


def print_epoch_banner(epoch: int, total: int, row: dict) -> None:
    """单行打印：进度条 | 轮次 | 损失 | 准确率指标。"""

    def f(key, default="-", nd=4):
        v = row.get(key, default)
        try:
            return f"{float(v):.{nd}f}"
        except Exception:
            return str(v) if v not in (None, "") else "-"

    pct = 100.0 * epoch / max(total, 1)
    bar_n = int(pct // 5)
    bar = "█" * bar_n + "░" * (20 - bar_n)
    box = f("train/box_loss")
    cls = f("train/cls_loss")
    dfl = f("train/dfl_loss")
    loss_sum = "-"
    try:
        loss_sum = f"{float(row.get('train/box_loss') or 0) + float(row.get('train/cls_loss') or 0) + float(row.get('train/dfl_loss') or 0):.4f}"
    except Exception:
        pass
    print(
        f"[{bar}] {epoch}/{total} ({pct:5.1f}%)  "
        f"loss={loss_sum} (box={box} cls={cls} dfl={dfl})  "
        f"P={f('metrics/precision(B)')} R={f('metrics/recall(B)')} "
        f"mAP50={f('metrics/mAP50(B)')} mAP50-95={f('metrics/mAP50-95(B)')}"
    )


def setup_matplotlib_chinese() -> None:
    import matplotlib

    matplotlib.use("Agg")  # 无界面环境也可保存图片
    import matplotlib.pyplot as plt

    # Windows 常见中文字体，按优先级尝试
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DengXian",
        "KaiTi",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"


def load_results_csv(save_dir: Path) -> list[dict]:
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({(k or "").strip(): v for k, v in r.items()})
    return rows


def _floats(rows: list[dict], *keys: str) -> list[float]:
    out = []
    for r in rows:
        val = None
        for k in keys:
            if k in r and r[k] not in (None, ""):
                val = r[k]
                break
        try:
            out.append(float(val))
        except Exception:
            out.append(float("nan"))
    return out


def visualize_training(save_dir: Path) -> None:
    """生成中文训练可视化图（损失 / 准确率相关指标 / 热力图）。"""
    import matplotlib.pyplot as plt
    import numpy as np

    setup_matplotlib_chinese()
    rows = load_results_csv(save_dir)
    vis_dir = save_dir / "viz_cn"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"[可视化] 无 results.csv: {save_dir}")
        return

    epochs = _floats(rows, "epoch", "epochs")
    box = _floats(rows, "train/box_loss")
    cls = _floats(rows, "train/cls_loss")
    dfl = _floats(rows, "train/dfl_loss")
    vbox = _floats(rows, "val/box_loss")
    vcls = _floats(rows, "val/cls_loss")
    vdfl = _floats(rows, "val/dfl_loss")
    prec = _floats(rows, "metrics/precision(B)")
    rec = _floats(rows, "metrics/recall(B)")
    map50 = _floats(rows, "metrics/mAP50(B)")
    map95 = _floats(rows, "metrics/mAP50-95(B)")
    lr = _floats(rows, "lr/pg0", "lr/pg1", "lr/pg2")

    # ---- 图1：损失曲线 ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(epochs, box, label="训练 box 损失", linewidth=1.8)
    ax[0].plot(epochs, cls, label="训练 cls 损失", linewidth=1.8)
    ax[0].plot(epochs, dfl, label="训练 dfl 损失", linewidth=1.8)
    ax[0].set_title("训练损失曲线（越低越好）")
    ax[0].set_xlabel("训练轮次 (Epoch)")
    ax[0].set_ylabel("损失 (Loss)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(epochs, vbox, label="验证 box 损失", linewidth=1.8)
    ax[1].plot(epochs, vcls, label="验证 cls 损失", linewidth=1.8)
    ax[1].plot(epochs, vdfl, label="验证 dfl 损失", linewidth=1.8)
    ax[1].set_title("验证损失曲线（越低越好）")
    ax[1].set_xlabel("训练轮次 (Epoch)")
    ax[1].set_ylabel("损失 (Loss)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()
    fig.suptitle("YOLO 训练过程 — 损失函数可视化", fontsize=13)
    fig.tight_layout()
    path1 = vis_dir / "01_损失曲线.png"
    fig.savefig(path1)
    plt.close(fig)
    print(f"[可视化] {path1.name}")

    # ---- 图2：准确率相关指标 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, prec, label="精确率 Precision", linewidth=2)
    ax.plot(epochs, rec, label="召回率 Recall", linewidth=2)
    ax.plot(epochs, map50, label="mAP@50", linewidth=2)
    ax.plot(epochs, map95, label="mAP@50-95", linewidth=2)
    ax.set_title("检测准确率相关指标曲线")
    ax.set_xlabel("训练轮次 (Epoch)")
    ax.set_ylabel("指标值 (0~1)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    path2 = vis_dir / "02_准确率指标.png"
    fig.savefig(path2)
    plt.close(fig)
    print(f"[可视化] {path2.name}")

    # ---- 图3：学习率曲线 ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lr, color="#d62728", linewidth=2)
    ax.set_title("学习率调度曲线")
    ax.set_xlabel("训练轮次 (Epoch)")
    ax.set_ylabel("学习率 (Learning Rate)")
    ax.grid(True, alpha=0.3)
    path3 = vis_dir / "03_学习率曲线.png"
    fig.savefig(path3)
    plt.close(fig)
    print(f"[可视化] {path3.name}")

    # ---- 图4：指标热力图（各 epoch 指标矩阵）----
    metric_names = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    matrix = np.array([prec, rec, map50, map95], dtype=float)
    # 若 epoch 太多，按列平均到最多 40 列便于阅读
    max_cols = 40
    if matrix.shape[1] > max_cols:
        idx = np.linspace(0, matrix.shape[1] - 1, max_cols).astype(int)
        matrix = matrix[:, idx]
        x_labels = [str(int(epochs[i])) for i in idx]
    else:
        x_labels = [str(int(e)) if e == e else "" for e in epochs]

    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("训练轮次 (Epoch)")
    ax.set_title("训练指标热力图（颜色越绿表示越好）")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("指标值")
    path4 = vis_dir / "04_指标热力图.png"
    fig.savefig(path4)
    plt.close(fig)
    print(f"[可视化] {path4.name}")

    # ---- 图5：混淆矩阵热力图（若 Ultralytics 已导出）----
    conf_path = save_dir / "confusion_matrix_normalized.png"
    if conf_path.exists():
        try:
            from PIL import Image

            img = Image.open(conf_path)
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.imshow(img)
            ax.set_title("混淆矩阵热力图（归一化）")
            ax.axis("off")
            path5 = vis_dir / "05_混淆矩阵热力图.png"
            fig.savefig(path5)
            plt.close(fig)
            print(f"[可视化] {path5.name}")
        except Exception as e:
            print(f"[可视化] 混淆矩阵跳过: {e}")

    # ---- 图6：综合看板 ----
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, box, label="box")
    ax1.plot(epochs, cls, label="cls")
    ax1.plot(epochs, dfl, label="dfl")
    ax1.set_title("训练损失")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, vbox, label="val box")
    ax2.plot(epochs, vcls, label="val cls")
    ax2.plot(epochs, vdfl, label="val dfl")
    ax2.set_title("验证损失")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, prec, label="Precision")
    ax3.plot(epochs, rec, label="Recall")
    ax3.plot(epochs, map50, label="mAP@50")
    ax3.plot(epochs, map95, label="mAP@50-95")
    ax3.set_title("准确率相关指标")
    ax3.set_xlabel("Epoch")
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    im = ax4.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax4.set_yticks(range(len(metric_names)))
    ax4.set_yticklabels(metric_names, fontsize=9)
    ax4.set_title("指标热力图")
    ax4.set_xlabel("抽样 Epoch")
    fig.colorbar(im, ax=ax4, fraction=0.046)

    fig.suptitle("YOLO 训练可视化综合看板（中文）", fontsize=14, y=0.98)
    path6 = vis_dir / "06_训练综合看板.png"
    fig.savefig(path6)
    plt.close(fig)
    print(f"[可视化] {path6.name}")

    # 摘要文本（不打印长文，只写文件 + 一行量化摘要）
    if map95 and map95[-1] == map95[-1]:
        best_i = int(np.nanargmax(map95))
        summary = (
            f"总轮次: {len(epochs)}\n"
            f"最优 mAP@50-95: {map95[best_i]:.4f}（第 {int(epochs[best_i])} 轮）\n"
            f"最终 Precision: {prec[-1]:.4f}  Recall: {rec[-1]:.4f}  mAP@50: {map50[-1]:.4f}\n"
            f"输出目录: {save_dir}\n"
        )
        (vis_dir / "训练摘要.txt").write_text(summary, encoding="utf-8")
        print(
            f"[摘要] epochs={len(epochs)} best_mAP50-95={map95[best_i]:.4f}@e{int(epochs[best_i])} "
            f"final P={prec[-1]:.4f} R={rec[-1]:.4f} mAP50={map50[-1]:.4f}"
        )


def train(args: argparse.Namespace) -> None:
    setup_utf8_console()
    setup_openmp_workaround()

    if os.environ.get("TRAIN_SKIP_ENV_SWITCH") != "1":
        ensure_cuda_pytorch(allow_cpu=args.allow_cpu, force_pytorch_env=args.force_pytorch_env)

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "当前环境缺少 torch/ultralytics，请在 conda pytorch 中 pip install ultralytics\n"
            f"原始错误: {exc}"
        ) from exc

    # 压制 Ultralytics 长日志（模型结构表、arg dump 等）
    try:
        import logging

        from ultralytics.utils import LOGGER

        LOGGER.setLevel(logging.ERROR)
    except Exception:
        pass

    gpu_ready = False
    gpu_name = "CPU"
    gpu_cap = ""
    torch_v = getattr(torch, "__version__", "?")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        gpu_cap = f"sm_{cap[0]}{cap[1]}"
        try:
            x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
            y = (x @ x).sum()
            torch.cuda.synchronize()
            gpu_ready = True
            _ = float(y)
        except Exception as e:
            gpu_ready = False
            print_sm120_fix_guide({"torch": torch_v, "error": str(e), "gpu": gpu_name, "capability": gpu_cap})

    if not gpu_ready and not args.allow_cpu:
        raise SystemExit("CUDA 不可用或 GPU kernel 缺失。可加 --allow-cpu，或升级 torch(cu128)。")

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"找不到数据配置: {data_yaml}")

    if args.subset and args.subset > 0:
        data_yaml = apply_subset(data_yaml, args.subset)

    device = args.device.strip() if args.device else (0 if gpu_ready else "cpu")
    dev_str = f"{gpu_name} {gpu_cap}" if gpu_ready else "CPU"

    # —— 启动摘要（仅量化关键信息）——
    print("-" * 72)
    print(f"设备   : {dev_str} | torch {torch_v} | device={device}")
    print(
        f"训练   : epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} "
        f"lr0={args.lr0} opt={args.optimizer} amp={args.amp} workers={args.workers}"
    )
    print(f"早停   : patience={args.patience} min_delta={args.min_delta} (mAP50-95)")
    print(f"数据   : {data_yaml}")
    print(f"输出   : {args.project}\\{args.name}")
    print("-" * 72)

    model = YOLO(args.model)
    monitor = EarlyStopMonitor(patience=args.patience, min_delta=args.min_delta)

    def on_fit_epoch_end(trainer):
        metrics = getattr(trainer, "metrics", None) or {}
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total = int(getattr(trainer, "epochs", args.epochs))
        row = {
            "epoch": epoch,
            "metrics/precision(B)": metrics.get("metrics/precision(B)", metrics.get("precision", "")),
            "metrics/recall(B)": metrics.get("metrics/recall(B)", metrics.get("recall", "")),
            "metrics/mAP50(B)": metrics.get("metrics/mAP50(B)", metrics.get("mAP50", "")),
            "metrics/mAP50-95(B)": metrics.get("metrics/mAP50-95(B)", metrics.get("mAP50-95", "")),
            "train/box_loss": "",
            "train/cls_loss": "",
            "train/dfl_loss": "",
        }
        try:
            tloss = getattr(trainer, "loss", None)
            if tloss is not None and hasattr(tloss, "tolist"):
                vals = tloss.tolist()
                if len(vals) >= 3:
                    row["train/box_loss"] = vals[0]
                    row["train/cls_loss"] = vals[1]
                    row["train/dfl_loss"] = vals[2]
        except Exception:
            pass
        print_epoch_banner(epoch, total, row)
        if monitor.update(row):
            print(
                f"[早停] epoch {epoch}: 连续{monitor.patience}轮 mAP50-95 提升不足 {monitor.min_delta}；"
                f"best={monitor.best:.4f} @ e{monitor.best_epoch}"
            )
            try:
                trainer.stop = True
            except Exception:
                pass

    try:
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    except Exception:
        pass

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        project=args.project,
        exist_ok=True,
        workers=args.workers,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        amp=bool(args.amp),
        patience=args.ultralytics_patience,
        pretrained=True,
        verbose=False,
        plots=True,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    print("-" * 72)
    print(f"完成   : best={best}")
    print(f"指标   : best_mAP50-95={monitor.best:.4f} @ epoch {monitor.best_epoch}")
    print(f"导出   : python training/export_onnx.py --weights \"{best}\" --name widerperson_person")
    print("-" * 72)

    if not args.no_vis:
        try:
            visualize_training(save_dir)
        except Exception as e:
            print(f"可视化失败: {e}")


def main() -> None:
    setup_utf8_console()
    setup_openmp_workaround()
    args = parse_args()

    # 仅可视化已有结果
    if args.save_dir_hint:
        vis_dir = Path(args.save_dir_hint)
        if not vis_dir.exists():
            raise SystemExit(f"目录不存在: {vis_dir}")
        setup_matplotlib_chinese()
        visualize_training(vis_dir)
        return

    train(args)


if __name__ == "__main__":
    main()
