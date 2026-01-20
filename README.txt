# YOLO铁路检测模型训练

本项目使用YOLO模型在铁路检测数据集上进行目标检测训练。

## 项目结构

```
多模态铁路检测/
├── Dataset/                    # 数据集目录
│   ├── data.yaml              # 数据集配置文件
│   ├── images/                # 图像目录
│   │   ├── train/            # 训练集图像
│   │   ├── val/              # 验证集图像
│   │   └── test/             # 测试集图像
│   └── labels/                # 标签目录
│       ├── train/            # 训练集标签
│       ├── val/              # 验证集标签
│       └── test/             # 测试集标签
├── train_yolo.py              # 训练脚本
├── requirements.txt           # Python依赖包
├── yolo11n.pt                 # 预训练模型权重
└── runs/                      # 训练结果输出目录（训练后生成）
    └── train/                 # 训练实验目录
        ├── weights/           # 模型权重
        │   ├── best.pt       # 最佳模型
        │   └── last.pt       # 最新模型
        ├── results.png        # 训练曲线图
        ├── confusion_matrix.png  # 混淆矩阵
        ├── train_params.yaml  # 训练参数
        └── evaluation_results.txt  # 评估结果
```

## 数据集信息

- **类别数量**: 6
- **类别名称**: 
  - person (人员)
  - obsticle_oc (障碍物)
  - Animal (动物)
  - vehicle (车辆)
  - motor_bicycle (摩托车)
  - Train (火车)

## 环境配置

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 2. 检查CUDA（如果使用GPU）

确保已安装CUDA和cuDNN（如果使用GPU训练）。可以通过以下命令检查：

```python
import torch
print(torch.cuda.is_available())
```

## 使用方法

### 训练模型

直接运行训练脚本：

```bash
python train_yolo.py
```

### 训练参数说明

训练脚本中的主要参数（可在`train_yolo.py`中修改）：

- `epochs`: 训练轮数（默认100）
- `batch`: 批次大小（默认16，根据GPU内存调整）
- `imgsz`: 输入图像尺寸（默认640）
- `device`: 设备选择（0表示GPU 0，'cpu'表示CPU）
- `lr0`: 初始学习率（默认0.01）
- `optimizer`: 优化器（auto, SGD, Adam, AdamW）

### 使用CPU训练

如果只有CPU，请在`train_yolo.py`中将：
```python
'device': 0,
```
改为：
```python
'device': 'cpu',
```

## 训练结果

训练完成后，所有结果将保存在`runs/train/`目录下：

1. **模型权重**
   - `weights/best.pt`: 验证集上表现最好的模型
   - `weights/last.pt`: 最后一轮的模型

2. **可视化结果**
   - `results.png`: 训练和验证指标曲线（损失、mAP等）
   - `confusion_matrix.png`: 混淆矩阵
   - `val_batch0_labels.jpg`: 验证批次可视化

3. **训练日志**
   - `train_params.yaml`: 训练参数配置
   - `evaluation_results.txt`: 模型评估结果
   - TensorBoard日志（如果启用）

4. **训练过程信息**
   - 每个epoch的训练损失、验证损失
   - mAP50、mAP50-95等指标
   - 学习率变化曲线

## 查看训练过程

### 使用TensorBoard

训练过程中会自动生成TensorBoard日志，可以通过以下命令查看：

```bash
tensorboard --logdir runs/train
```

然后在浏览器中打开 `http://localhost:6006`

## 模型评估

训练脚本会自动在验证集上评估模型，评估结果包括：

- **mAP50**: 在IoU=0.5时的平均精度
- **mAP50-95**: 在IoU=0.5-0.95时的平均精度
- **精确度**: Precision
- **召回率**: Recall

## 使用训练好的模型

### 加载最佳模型进行推理

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/weights/best.pt')

# 进行预测
results = model('path/to/image.jpg')

# 显示结果
results[0].show()
```

## 注意事项

1. **GPU内存**: 如果遇到GPU内存不足，可以减小`batch`大小或`imgsz`
2. **训练时间**: 训练时间取决于数据集大小、GPU性能和训练轮数
3. **数据路径**: 确保`Dataset/data.yaml`中的路径正确
4. **标签格式**: 标签文件应为YOLO格式（每行：class_id x_center y_center width height，归一化坐标）

## 故障排除

### 常见问题

1. **CUDA out of memory**
   - 减小`batch`大小
   - 减小`imgsz`
   - 使用更小的模型（如yolo11n.pt）

2. **找不到数据集**
   - 检查`Dataset/data.yaml`中的路径是否正确
   - 确保图像和标签文件存在

3. **训练速度慢**
   - 使用GPU而非CPU
   - 增加`workers`数量（但不要超过CPU核心数）
   - 使用混合精度训练（`amp=True`）

## 许可证

本项目仅供学习和研究使用。




