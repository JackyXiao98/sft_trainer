import os
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset, Dataset

# Define the base directory and list of datasets
base_dir = '/mnt/hdfs/selection/from_jiaxiang_wu/general_n_safety_datasets'
datasets_to_load = {
    "通用任务, non-thinking模式训练数据 (5 domains)": [
        'tulu3_qwen3_2507_no_think_coding',
        'tulu3_qwen3_2507_no_think_instruction',
        'tulu3_qwen3_2507_no_think_knowledge',
        'tulu3_qwen3_2507_no_think_math',
        'tulu3_qwen3_2507_no_think_multilingual'
    ],
    "通用任务, thinking模式训练数据 (4 domains)": [
        'open_r1_qwen3_2507_0804_think_coding_8k',
        'open_r1_qwen3_2507_0804_think_math_8k',
        'tulu3_qwen3_2507_0805_think_knowledge_8k',
        'tulu3_qwen3_2507_0805_think_multilingual_8k'
    ],
    "安全对齐任务, 中英文为主的non-thinking & thinking模式训练数据 (4 domains)": [
        'safety_cn_bias',
        'safety_tier1',
        'safety_tier2',
        'safety_tier3'
    ]
}

def load_parquet_safely(file_path):
    """安全地加载parquet文件，处理元数据问题"""
    try:
        # 方法1: 直接使用datasets库加载
        dataset = load_dataset('parquet', data_files=file_path)
        return dataset, "datasets"
    except Exception as e1:
        print(f"  ⚠️  datasets库加载失败: {e1}")
        try:
            # 方法2: 使用pandas读取，然后转换为Dataset
            print("  🔄 尝试使用pandas读取...")
            df = pd.read_parquet(file_path)
            dataset = Dataset.from_pandas(df)
            return {"train": dataset}, "pandas"
        except Exception as e2:
            print(f"  ⚠️  pandas加载失败: {e2}")
            try:
                # 方法3: 使用pyarrow直接读取
                print("  🔄 尝试使用pyarrow读取...")
                table = pq.read_table(file_path)
                df = table.to_pandas()
                dataset = Dataset.from_pandas(df)
                return {"train": dataset}, "pyarrow"
            except Exception as e3:
                print(f"  ❌ pyarrow加载失败: {e3}")
                return None, "failed"

# Iterate through each category and dataset
for category, file_list in datasets_to_load.items():
    print(f"\n{'='*60}")
    print(f"Category: {category}")
    print(f"{'='*60}")
    
    for file_name in file_list:
        file_path = os.path.join(base_dir, f'{file_name}.parquet')
        
        print(f"\n📁 文件路径: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"  ❌ 文件不存在: {file_path}")
            continue
        
        # 获取文件大小
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  📊 文件大小: {file_size:.2f} MB")
        
        # 安全加载数据集
        dataset, method = load_parquet_safely(file_path)
        
        if dataset is None:
            print(f"  ❌ 无法加载数据集: {file_name}")
            continue
        
        print(f"  ✅ 成功加载 (方法: {method})")
        print(f"\n--- Dataset: {file_name} ---")
        
        # Print column names and schema
        print("📋 列名:")
        train_dataset = dataset['train']
        print(f"  {train_dataset.column_names}")
        
        print(f"\n📊 数据集信息:")
        print(f"  样本数量: {len(train_dataset):,}")
        
        # 尝试打印特征信息
        try:
            print(f"\n🔍 特征类型:")
            for col_name in train_dataset.column_names:
                col_type = type(train_dataset[0][col_name]).__name__
                print(f"  {col_name}: {col_type}")
        except Exception as e:
            print(f"  ⚠️  无法获取特征类型: {e}")
        
        # Print the first example
        print(f"\n📝 第一个样本:")
        try:
            first_example = train_dataset[0]
            for key, value in first_example.items():
                # 限制显示长度
                if isinstance(value, str) and len(value) > 200:
                    display_value = value[:200] + "..."
                else:
                    display_value = value
                print(f"  {key}: {display_value}")
        except Exception as e:
            print(f"  ⚠️  无法显示第一个样本: {e}")
        
        print(f"\n{'─'*40}")
        
        # 添加交互式断点
        response = input("按 Enter 继续下一个数据集，输入 'q' 退出，输入 's' 跳过当前类别: ")
        if response.lower() == 'q':
            print("退出程序")
            exit()
        elif response.lower() == 's':
            print(f"跳过类别: {category}")
            break
            
