# 训练脚本位置与使用说明

## 脚本位置

| 脚本 | 路径 | 作用 |
|------|------|------|
| **正式训练（推荐）** | `D:\OpenCV\develop\1\training\train_yolov8.py` | 自动定位 conda `pytorch` 环境 CUDA、低学习率、高轮数、中文终端进度 + 训练可视化 |
| 调试入口（默认不训） | `D:\OpenCV\develop\1\training\train_person_cpu.py` | 打印推荐命令；加 `--run` 才会转发执行 |
| 数据转换 | `D:\OpenCV\develop\1\training\coco_to_yolo.py` | WiderPerson COCO → YOLO |
| ONNX 导出 | `D:\OpenCV\develop\1\training\export_person_onnx.py` | person 模型导出到 `models\` |
| ONNX 导出（通用） | `D:\OpenCV\develop\1\training\export_onnx.py` | 任意 best.pt 导出 |

## 请自行训练（不要由本机助手代跑）

打开终端（PowerShell）：

```powershell
# 1) 优先使用 conda「pytorch」环境（CUDA）
#    脚本也会自动探测并切换到该环境
D:\Anaconda\envs\pytorch\python.exe D:\OpenCV\develop\1\training\train_yolov8.py --data D:\OpenCV\develop\1\data\widerperson_yolo\data.yaml --model D:\OpenCV\develop\1\models\yolov8n.pt --epochs 100 --batch 16 --lr0 0.001 --patience 25 --name widerperson_person_gpu
```

### 参数默认值（已按你的要求调整）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--epochs` | **100** | 已提高 |
| `--lr0` | **0.001** | 已降低（原先 auto 约 0.002） |
| `--lrf` | 0.01 | 最终学习率比例 |
| `--patience` | **25** | 自定义早停（监控 mAP50-95 + min_delta） |
| `--min-delta` | 0.002 | 认为“有提升”的最小幅度 |
| `--ultralytics-patience` | 50 | 放宽内置早停，真正早停交给脚本 |

### 终端会打印

- 环境探测（当前解释器 / conda pytorch / GPU / sm 能力）
- 每轮进度条 + 中文指标（损失、P/R、mAP、学习率）
- 早停触发原因与历史最优

### 训练后可视化（中文标题）

输出目录：`runs\detect\<name>\viz_cn\`

- `01_损失曲线.png`
- `02_准确率指标.png`
- `03_学习率曲线.png`
- `04_指标热力图.png`
- `05_混淆矩阵热力图.png`（若已有 confusion matrix）
- `06_训练综合看板.png`
- `训练摘要.txt`

仅对已有结果出图：

```powershell
D:\Anaconda\envs\pytorch\python.exe D:\OpenCV\develop\1\training\train_yolov8.py --save-dir-hint D:\OpenCV\develop\1\runs\detect\widerperson_person
```

## RTX 5070 (sm_120) 注意

**已修复（本机）**：conda `pytorch` 环境已升级为 `torch 2.11.0+cu128`，GPU 冒烟测试通过。

若换机或重装后再次报错：

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

说明 torch 不含 sm_120 kernel，需执行：

```powershell
D:\Anaconda\envs\pytorch\python.exe -m pip uninstall -y torch torchvision
D:\Anaconda\envs\pytorch\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 脚本针对你上次报错的修复

| 报错/现象 | 处理 |
|-----------|------|
| `no kernel image is available` | 启动前做 **GPU 算子冒烟测试**；失败则打印修复指南并退出（可 `--allow-cpu`） |
| AMP 去 GitHub 下载 `yolo26n.pt` 失败 | **默认 `amp=False`**；需要时加 `--amp` |
| `optimizer=auto` 忽略 `lr0=0.001` | 默认 **`optimizer=SGD`**，学习率真正生效 |
| `OMP Error #15 libiomp5md.dll` | 自动设置 `KMP_DUPLICATE_LIB_OK=TRUE`、`OMP_NUM_THREADS=1` |
| `DataLoader worker exited unexpectedly` | 默认 **`--workers 0`**（主进程加载，最稳）；可试 `--workers 2` |

## 推荐训练命令（请自行在终端执行）

```powershell
chcp 65001
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="1"
D:\Anaconda\envs\pytorch\python.exe D:\OpenCV\develop\1\training\train_yolov8.py --data D:\OpenCV\develop\1\data\widerperson_yolo\data.yaml --model D:\OpenCV\develop\1\models\yolov8n.pt --epochs 100 --batch 16 --lr0 0.001 --patience 25 --workers 0 --name widerperson_person_gpu
```

Windows 若 pip 报 Access denied，可改用批处理：`training\install_cu128.bat`

## 中文编码

- 脚本已设置 `PYTHONUTF8=1`、`stdout/stderr` 为 UTF-8
- matplotlib 使用 `Microsoft YaHei` / `SimHei`，并关闭 `axes.unicode_minus`
- 若终端仍乱码，可在 PowerShell 执行：`chcp 65001`
