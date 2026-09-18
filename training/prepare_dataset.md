# 自定义数据集准备（YOLO 格式）

## 目录

```text
dataset/
├─ images/train/   # 训练图 jpg/png
├─ images/val/     # 验证图
├─ labels/train/   # 与图片同名的 .txt
├─ labels/val/
└─ data.yaml
```

## 标签格式

每张图一个 txt，每行一个目标：

```text
<class_id> <cx> <cy> <w> <h>
```

全部归一化到 `0~1`。例如图片 1280×720，目标中心 (640,360)、宽高 200×100：

```text
0 0.5 0.5 0.15625 0.138889
```

## data.yaml

```yaml
path: D:/OpenCV/develop/1/dataset
train: images/train
val: images/val
names:
  0: person
  1: helmet
  2: forklift
```

`names` 的顺序必须与导出的 `classes.txt` 一致，否则界面类别会错位。

## 标注工具（任选其一）

- [Roboflow](https://roboflow.com)（导出 YOLO）
- [Label Studio](https://labelstud.io)
- [LabelImg](https://github.com/HumanSignal/labelImg) / X-AnyLabeling
- CVAT

## 建议

- 每类 ≥ 100–300 张再训，小样本优先数据增强
- 场景与摄像头预览尽量一致（分辨率、光照、角度）
- 训练输入 `imgsz=640`，与本程序默认一致
