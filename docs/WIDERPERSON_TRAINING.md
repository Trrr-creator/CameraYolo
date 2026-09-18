# WiderPerson → CameraYolo

本目录已用你提供的 `widerperson_coco.zip`（COO 格式 WiderPerson，单类 pedestrian/person）完成转换、训练与 ONNX 导出。

## 数据

| 路径 | 说明 |
|------|------|
| `data/widerperson_coco.zip` | 原始包（train 8000 / val 1000 / test 4382） |
| `data/widerperson_coco/` | 解压后的 COCO json + 图片 |
| `data/widerperson_yolo/` | 完整 YOLO 格式（8000 train + 1000 val） |
| `data/widerperson_yolo_subset/` | CPU 快训子集（2000 train + 300 val） |

## 训练

- 环境：`D:\Anaconda\envs\opencv\python.exe`（已装 ultralytics 8.4 + torch CPU）
- 模型：`yolov8n.pt` 预训练，head 改为 1 类 `person`
- 脚本：`training/train_person_cpu.py`
- 输出：`runs/detect/widerperson_person/`

```powershell
# CPU 快训（当前配置）
D:\Anaconda\envs\opencv\python.exe training\train_person_cpu.py --max-train 2000 --epochs 12 --batch 8 --imgsz 640 --device cpu

# 全量 8000 张（建议 GPU）
D:\Anaconda\envs\opencv\python.exe training\train_yolov8.py --data data\widerperson_yolo\data.yaml --epochs 50 --batch 16 --imgsz 640 --name widerperson_full
```

## 导出 ONNX（已生成）

```powershell
D:\Anaconda\envs\opencv\python.exe training\export_person_onnx.py
```

产物：

- `models/widerperson_person.onnx`（约 12MB，输入 1×3×640×640，输出 1×5×8400）
- `models/classes.txt` → `person`

## 在 CameraYolo 中使用

1. 启动 `run.ps1` 或 `CameraYolo\bin\Release\net8.0-windows\CameraYolo.exe`
2. 「加载模型…」→ 选择 `D:\OpenCV\develop\1\models\widerperson_person.onnx`
3. 类别文件会自动读同目录 `classes.txt`
4. 选摄像头 → 启动

## 说明

- WiderPerson 官网在本机不可访问；你手动下载的 zip 结构正确，已直接使用。
- GPU 版 torch（cu128）因镜像限速未装完，当前用 CPU 训练。完整训练建议：
  1. 从 `https://mirrors.aliyun.com/pytorch-wheels/cu128/` 下载 `torch-2.10.0+cu128-cp311-win_amd64.whl`
  2. `pip install <whl> torchvision`
  3. `--device 0` 全量训练
