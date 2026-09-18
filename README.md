# CameraYolo — 摄像头实时 YOLO 目标检测

基于 C# WPF + OpenCvSharp（底层 C++ OpenCV DNN）的摄像头实时目标检测程序。支持加载自训 YOLO 模型（ONNX/Darknet）。

## 功能特性

- 实时摄像头预览 + YOLO 目标检测
- 支持 ONNX（YOLOv5/v8/v11）和 Darknet（YOLOv3/v4）模型
- 自动检测推理设备（CUDA / OpenCL / CPU）
- 置信度阈值调节
- 自动读取类别文件（classes.txt / coco.names / labels.txt）

## 环境要求

- Windows 10/11 x64
- .NET 8 SDK
- 摄像头
- （可选）NVIDIA GPU + CUDA 加速推理

## 快速开始

```powershell
# 编译
dotnet build -c Release

# 运行
.\CameraYolo\bin\Release\net8.0-windows\CameraYolo.exe
```

## 使用说明

1. 点击 **「加载模型」** 选择 ONNX 或 Darknet 模型文件
2. 在设备下拉框选择推理设备（默认自动检测）
3. 选择摄像头编号 → 点击 **「启动」**
4. 拖动置信度滑条调整检测灵敏度

## 项目结构

```
├─ CameraYolo/              # WPF 应用源码
│  ├─ MainWindow.xaml(.cs)  # 主界面
│  ├─ Services/
│  │  ├─ CameraPipeline.cs  # 摄像头采集管线
│  │  └─ YoloDetector.cs    # YOLO 推理引擎
│  └─ Models/               # 数据模型
├─ models/                  # 预置模型
│  ├─ widerperson_person.onnx
│  ├─ yolov8n.pt
│  └─ classes.txt
└─ training/                # 训练脚本（可选）
   ├─ train_yolov8.py       # YOLOv8 训练
   └─ export_onnx.py        # 导出 ONNX
```

## 自训模型

```powershell
# 1. 安装依赖
pip install ultralytics

# 2. 训练
python training\train_yolov8.py --data dataset\data.yaml --epochs 80 --imgsz 640

# 3. 导出 ONNX
python training\export_onnx.py --weights runs\detect\train\weights\best.pt

# 4. 加载模型
# 将生成的 .onnx 和 classes.txt 放入 models/ 目录
```

## 许可证

MIT License