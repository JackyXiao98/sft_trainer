#!/usr/bin/env python3
"""
测试数据格式和预处理功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import SFTTrainingPipeline
import argparse

def test_data_format(dataset_path, config_path="configs/training_config.yaml"):
    """测试数据格式"""
    print(f"测试数据集: {dataset_path}")
    print(f"使用配置: {config_path}")
    
    try:
        # 创建训练管道
        pipeline = SFTTrainingPipeline(config_path)
        
        # 只加载和预处理数据集，不进行训练
        print("\n=== 加载数据集 ===")
        dataset = pipeline.load_dataset(dataset_path)
        
        print(f"\n=== 数据集信息 ===")
        print(f"数据集大小: {len(dataset)}")
        print(f"数据集字段: {list(dataset[0].keys())}")
        
        # 检查前几个样本
        print(f"\n=== 样本检查 ===")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            text = sample.get('text', '')
            print(f"样本 {i}:")
            print(f"  - 类型: {type(text)}")
            print(f"  - 长度: {len(text) if isinstance(text, str) else 'N/A'}")
            print(f"  - 预览: {str(text)[:100]}...")
            print()
        
        print("✅ 数据格式测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 数据格式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试数据格式")
    parser.add_argument("dataset_path", help="数据集路径")
    parser.add_argument("--config", default="configs/training_config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    success = test_data_format(args.dataset_path, args.config)
    sys.exit(0 if success else 1)