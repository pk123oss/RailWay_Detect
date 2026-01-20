"""
数据集检查脚本
用于验证数据集配置是否正确，图像和标签是否匹配
"""

import os
from pathlib import Path
import yaml

def check_dataset():
    """检查数据集配置和文件"""
    project_root = Path(__file__).parent
    dataset_yaml = project_root / "Dataset" / "data.yaml"
    
    print("=" * 60)
    print("数据集检查")
    print("=" * 60)
    
    # 检查配置文件
    if not dataset_yaml.exists():
        print(f"❌ 错误: 数据集配置文件不存在: {dataset_yaml}")
        return False
    
    print(f"✅ 找到配置文件: {dataset_yaml}")
    
    # 读取配置
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\n数据集配置:")
    print(f"  类别数量: {data_config.get('nc', 'N/A')}")
    print(f"  类别名称: {data_config.get('names', 'N/A')}")
    
    # 检查各个数据集路径
    splits = ['train', 'val', 'test']
    all_ok = True
    
    for split in splits:
        print(f"\n检查 {split} 集:")
        
        # 获取图像和标签路径
        img_path_str = data_config.get(split, '')
        if not img_path_str:
            print(f"  ❌ 错误: {split} 路径未配置")
            all_ok = False
            continue
        
        # 处理相对路径（相对于data.yaml文件所在目录）
        if os.path.isabs(img_path_str):
            img_dir = Path(img_path_str)
        else:
            # 路径相对于Dataset目录（data.yaml所在目录）
            img_dir = dataset_yaml.parent / img_path_str
        
        # 标签目录在labels文件夹下，与images同级
        label_dir = dataset_yaml.parent / "labels" / split
        
        # 检查图像目录
        if not img_dir.exists():
            print(f"  ❌ 错误: 图像目录不存在: {img_dir}")
            all_ok = False
            continue
        
        # 检查标签目录
        if not label_dir.exists():
            print(f"  ❌ 错误: 标签目录不存在: {label_dir}")
            all_ok = False
            continue
        
        # 统计文件数量
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        label_files = list(label_dir.glob("*.txt"))
        
        print(f"  ✅ 图像目录: {img_dir}")
        print(f"  ✅ 标签目录: {label_dir}")
        print(f"  📊 图像数量: {len(img_files)}")
        print(f"  📊 标签数量: {len(label_files)}")
        
        # 检查图像和标签是否匹配
        img_names = {f.stem for f in img_files}
        label_names = {f.stem for f in label_files}
        
        missing_labels = img_names - label_names
        missing_images = label_names - img_names
        
        if missing_labels:
            print(f"  ⚠️  警告: {len(missing_labels)} 个图像没有对应的标签文件")
            if len(missing_labels) <= 5:
                for name in list(missing_labels)[:5]:
                    print(f"      - {name}")
        
        if missing_images:
            print(f"  ⚠️  警告: {len(missing_images)} 个标签文件没有对应的图像")
            if len(missing_images) <= 5:
                for name in list(missing_images)[:5]:
                    print(f"      - {name}")
        
        if not missing_labels and not missing_images:
            print(f"  ✅ 图像和标签完全匹配")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ 数据集检查完成，可以开始训练！")
    else:
        print("❌ 数据集检查发现问题，请修复后再训练")
    print("=" * 60)
    
    return all_ok

if __name__ == "__main__":
    check_dataset()

