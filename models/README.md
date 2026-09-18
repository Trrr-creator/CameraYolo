# 模型目录

将 YOLO 模型放在这里：

- `*.onnx` — 推荐（YOLOv5 / YOLOv8 / YOLOv11 / 自训导出）
- `classes.txt` — 每行一个类别名，顺序与训练 `data.yaml` 的 names 一致

## 获取预训练 COCO 模型（可选）

本机网络若无法直接访问 GitHub，可任选其一：

1. 在能联网的机器下载 `yolov8n.onnx`，拷入本目录
2. 本机安装 ultralytics 后导出：
   ```powershell
   pip install ultralytics
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640)"
   move yolov8n.onnx models\
   ```
3. 使用自己训练的模型，见 `../training/`

程序启动后点「加载模型…」选择本目录中的 `.onnx` 即可。
