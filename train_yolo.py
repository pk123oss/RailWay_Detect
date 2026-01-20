"""
YOLO模型训练脚本
用于在铁路检测数据集上训练YOLO模型
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch

def train_yolo_model():
    """
    训练YOLO模型并保存所有训练结果
    """
    # 设置项目路径
    project_root = Path(__file__).parent
    dataset_yaml = project_root / "Dataset" / "data.yaml"
    
    # 创建输出目录
    runs_dir = project_root / "runs"
    runs_dir.mkdir(exist_ok=True)
    
    # 检查数据集配置文件
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {dataset_yaml}")
    
    # 读取数据集配置
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    print("=" * 60)
    print("YOLO模型训练配置")
    print("=" * 60)
    print(f"数据集配置: {dataset_yaml}")
    print(f"类别数量: {data_config.get('nc', 'N/A')}")
    print(f"类别名称: {data_config.get('names', 'N/A')}")
    
    # 自动检测设备
    if torch.cuda.is_available():
        device = 0
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        device = 'cpu'
        print("未检测到GPU，将使用CPU训练（速度较慢）")
    print("=" * 60)
    
    # 初始化模型（使用预训练模型或从头开始）
    model_path = project_root / "yolo11n.pt"
    if model_path.exists():
        print(f"使用预训练模型: {model_path}")
        model = YOLO(str(model_path))
    else:
        print("使用YOLOv11n默认预训练模型")
        model = YOLO('yolo11n.pt')  # 会自动下载
    
    # 训练参数配置
    train_args = {
        'data': str(dataset_yaml),           # 数据集配置文件路径
        'epochs': 100,                       # 训练轮数
        'imgsz': 640,                        # 输入图像尺寸
        'batch': 16 if torch.cuda.is_available() else 4,  # 批次大小（GPU用16，CPU用4）
        'device': device,                    # 自动检测的设备
        'workers': 8,                        # 数据加载线程数
        'project': str(runs_dir),            # 项目保存目录
        'name': 'train',                     # 实验名称
        'exist_ok': True,                    # 允许覆盖已存在的实验
        'pretrained': True,                  # 使用预训练权重
        'optimizer': 'auto',                 # 优化器：auto, SGD, Adam, AdamW
        'verbose': True,                     # 详细输出
        'seed': 42,                          # 随机种子
        'deterministic': True,               # 确定性训练
        'single_cls': False,                 # 单类别模式
        'rect': False,                       # 矩形训练
        'cos_lr': False,                     # 余弦学习率调度
        'close_mosaic': 10,                  # 最后N个epoch关闭mosaic增强
        'resume': False,                     # 是否从检查点恢复训练
        'amp': True,                         # 自动混合精度训练
        'fraction': 1.0,                     # 使用数据集的比例
        'profile': False,                    # 性能分析
        'freeze': None,                      # 冻结层数
        'lr0': 0.01,                         # 初始学习率
        'lrf': 0.01,                         # 最终学习率（lr0 * lrf）
        'momentum': 0.937,                   # SGD动量
        'weight_decay': 0.0005,              # 权重衰减
        'warmup_epochs': 3.0,                # 预热轮数
        'warmup_momentum': 0.8,              # 预热动量
        'warmup_bias_lr': 0.1,               # 预热偏置学习率
        'box': 7.5,                          # 边界框损失权重
        'cls': 0.5,                          # 分类损失权重
        'dfl': 1.5,                          # DFL损失权重
        'pose': 12.0,                        # 姿态损失权重（如果适用）
        'kobj': 1.0,                         # 关键点对象损失权重
        'label_smoothing': 0.0,              # 标签平滑
        'nbs': 64,                           # 标称批次大小
        'hsv_h': 0.015,                      # 色调增强
        'hsv_s': 0.7,                        # 饱和度增强
        'hsv_v': 0.4,                        # 明度增强
        'degrees': 0.0,                      # 旋转角度
        'translate': 0.1,                    # 平移
        'scale': 0.5,                        # 缩放
        'shear': 0.0,                        # 剪切
        'perspective': 0.0,                  # 透视变换
        'flipud': 0.0,                       # 上下翻转概率
        'fliplr': 0.5,                       # 左右翻转概率
        'mosaic': 1.0,                       # 马赛克增强概率
        'mixup': 0.0,                        # 混合增强概率
        'copy_paste': 0.0,                   # 复制粘贴增强概率
    }
    
    # 开始训练
    print("\n开始训练...")
    print("-" * 60)
    
    try:
        # 执行训练
        results = model.train(**train_args)
        
        # 训练完成后的信息
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        
        # 获取最佳模型路径
        best_model_path = runs_dir / "train" / "weights" / "best.pt"
        last_model_path = runs_dir / "train" / "weights" / "last.pt"
        
        print(f"\n训练结果保存在: {runs_dir / 'train'}")
        print(f"最佳模型: {best_model_path}")
        print(f"最新模型: {last_model_path}")
        print(f"训练日志: {runs_dir / 'train'}")
        print(f"可视化结果: {runs_dir / 'train' / 'results.png'}")
        print(f"混淆矩阵: {runs_dir / 'train' / 'confusion_matrix.png'}")
        print(f"验证曲线: {runs_dir / 'train' / 'val_batch0_labels.jpg'}")
        
        # 保存训练参数到文件
        params_file = runs_dir / "train" / "train_params.yaml"
        with open(params_file, 'w', encoding='utf-8') as f:
            yaml.dump(train_args, f, allow_unicode=True, default_flow_style=False)
        print(f"训练参数已保存: {params_file}")
        
        # 在验证集上评估最佳模型
        print("\n在验证集上评估最佳模型...")
        metrics = model.val(data=str(dataset_yaml))
        
        print("\n验证指标:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"精确度: {metrics.box.mp:.4f}")
        print(f"召回率: {metrics.box.mr:.4f}")
        
        # 保存评估结果
        eval_file = runs_dir / "train" / "evaluation_results.txt"
        with open(eval_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("模型评估结果\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"mAP50: {metrics.box.map50:.4f}\n")
            f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
            f.write(f"精确度: {metrics.box.mp:.4f}\n")
            f.write(f"召回率: {metrics.box.mr:.4f}\n")
        print(f"评估结果已保存: {eval_file}")
        
        return results, best_model_path
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    train_yolo_model()

