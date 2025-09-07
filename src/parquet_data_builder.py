#!/usr/bin/env python3
"""
Parquet数据集构建器
用于处理read_dataset.py中定义的13个parquet数据集，生成53个训练集（1 + 13*4）
"""

import os
import yaml
import random
import pandas as pd
import pyarrow.parquet as pq
from typing import List, Dict, Tuple
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from rich.console import Console
from rich.progress import Progress, TaskID

from data_builder import DataBuilder

console = Console()

class ParquetDataBuilder(DataBuilder):
    """处理parquet格式数据集的DataBuilder子类"""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        super().__init__(config_path)
        
        # parquet数据集配置
        self.base_dir = '/mnt/hdfs/selection/from_jiaxiang_wu/general_n_safety_datasets'
        self.datasets_to_load = {
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
        
        # 获取所有数据集名称列表
        self.all_dataset_names = []
        for category, datasets in self.datasets_to_load.items():
            self.all_dataset_names.extend(datasets)
        
        console.print(f"[green]发现 {len(self.all_dataset_names)} 个parquet数据集[/green]")
        console.print(f"将生成 {1 + len(self.all_dataset_names) * 4} 个训练集")
    
    def load_parquet_safely(self, file_path: str) -> Tuple[Dataset, str]:
        """安全加载parquet文件，支持多种方法"""
        console.print(f"  🔄 正在加载: {os.path.basename(file_path)}")
        
        # 方法1: 使用datasets库
        try:
            dataset = load_dataset('parquet', data_files=file_path)
            return dataset, "datasets"
        except Exception as e:
            console.print(f"  ⚠️  datasets方法失败: {e}")
        
        # 方法2: 使用pandas
        try:
            df = pd.read_parquet(file_path)
            dataset = Dataset.from_pandas(df)
            return {"train": dataset}, "pandas"
        except Exception as e:
            console.print(f"  ⚠️  pandas方法失败: {e}")
        
        # 方法3: 使用pyarrow
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            dataset = Dataset.from_pandas(df)
            return {"train": dataset}, "pyarrow"
        except Exception as e:
            console.print(f"  ⚠️  pyarrow方法失败: {e}")
        
        return None, "failed"
    
    def load_source_datasets(self) -> Dict[str, Dataset]:
        """加载parquet源数据集"""
        console.print("[blue]正在加载parquet数据集...[/blue]")
        
        datasets = {}
        
        for category, file_list in self.datasets_to_load.items():
            console.print(f"\n[cyan]类别: {category}[/cyan]")
            
            for file_name in file_list:
                file_path = os.path.join(self.base_dir, f'{file_name}.parquet')
                
                if not os.path.exists(file_path):
                    console.print(f"  ❌ 文件不存在: {file_path}")
                    continue
                
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                console.print(f"  📊 文件大小: {file_size:.2f} MB")
                
                dataset_result, method = self.load_parquet_safely(file_path)
                
                if dataset_result is None:
                    console.print(f"  ❌ 无法加载数据集: {file_name}")
                    continue
                
                console.print(f"  ✅ 成功加载 (方法: {method})")
                
                # 获取训练数据集
                train_dataset = dataset_result['train']
                console.print(f"  📊 样本数量: {len(train_dataset):,}")
                
                # 存储数据集
                datasets[file_name] = train_dataset
        
        console.print(f"\n[green]成功加载 {len(datasets)} 个数据集[/green]")
        return datasets
    
    def create_base_datasets(self, source_datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """创建基础数据集（每个源数据集一个）"""
        console.print("[blue]创建基础数据集...[/blue]")
        
        base_datasets = {}
        
        console.print(f"[blue]开始处理 {len(source_datasets)} 个数据集...[/blue]")
        
        for i, (dataset_name, dataset) in enumerate(source_datasets.items(), 1):
            console.print(f"[cyan]({i}/{len(source_datasets)}) 处理数据集: {dataset_name}[/cyan]")
            
            # 按token限制采样（内部会显示进度）
            sampled_dataset = self.sample_by_token_count(
                dataset, self.train_token_limit, dataset_name
            )
            
            base_datasets[dataset_name] = sampled_dataset
            console.print(f"[green]✓ 完成数据集 {dataset_name} 的处理[/green]")
        
        console.print(f"[blue]所有数据集处理完成！[/blue]")
        return base_datasets
    
    def create_validation_datasets(self, source_datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """创建验证数据集（从13个数据集中采样）"""
        console.print("[blue]创建验证数据集...[/blue]")
        
        validation_datasets = {}
        
        console.print(f"[blue]开始为 {len(source_datasets)} 个数据集创建验证集...[/blue]")
        
        for i, (dataset_name, dataset) in enumerate(source_datasets.items(), 1):
            console.print(f"[cyan]({i}/{len(source_datasets)}) 为 {dataset_name} 创建验证集[/cyan]")
            
            # 按token限制采样验证集（内部会显示进度）
            val_dataset = self.sample_by_token_count(
                dataset, self.val_token_limit, f"{dataset_name}_val"
            )
            
            validation_datasets[f"{dataset_name}_val"] = val_dataset
            console.print(f"[green]✓ 完成验证集 {dataset_name}_val 的创建[/green]")
        
        console.print(f"[blue]所有验证集创建完成！[/blue]")
        return validation_datasets
    
    def create_training_variants(self, base_datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """创建训练变体：1个全量 + 13*4个变体扰动"""
        console.print("[blue]创建训练变体...[/blue]")
        
        training_datasets = {}
        dataset_names = list(base_datasets.keys())
        
        # 1. 创建全量数据集（所有13个数据集的组合）
        console.print("创建全量训练数据集...")
        all_datasets = list(base_datasets.values())
        if all_datasets:
            full_dataset = concatenate_datasets(all_datasets)
            training_datasets["full_dataset"] = full_dataset
            console.print(f"[green]全量数据集创建完成: {len(full_dataset):,} 样本[/green]")
        
        # 2. 为每个数据集创建4种变体扰动（1/3, 1/2, 2x, 3x）+ 所有其他数据集
        console.print(f"\n开始为 {len(dataset_names)} 个数据集创建变体扰动...")
        
        for i, target_dataset_name in enumerate(dataset_names, 1):
            console.print(f"\n[cyan]({i}/{len(dataset_names)}) 为 {target_dataset_name} 创建4种变体扰动[/cyan]")
            
            target_dataset = base_datasets[target_dataset_name]
            
            # 获取所有其他数据集
            other_datasets = [base_datasets[name] for name in dataset_names if name != target_dataset_name]
            
            # 合并所有其他数据集
            if other_datasets:
                other_combined = concatenate_datasets(other_datasets)
            else:
                # 如果没有其他数据集，使用空数据集
                other_combined = Dataset.from_dict({})
            
            # 创建4种变体扰动
            variants = self._create_dataset_variants(target_dataset, other_combined, target_dataset_name)
            training_datasets.update(variants)
        
        console.print(f"\n[green]创建了 {len(training_datasets)} 个训练数据集[/green]")
        return training_datasets
    
    def _create_dataset_variants(self, target_dataset: Dataset, other_combined: Dataset, dataset_name: str) -> Dict[str, Dataset]:
        """为特定数据集创建4个变体扰动"""
        variants = {}
        
        # 1/3 变体：目标数据集的1/3 + 所有其他数据集
        if len(target_dataset) >= 3:
            subset_1_3 = target_dataset.select(random.sample(range(len(target_dataset)), len(target_dataset) // 3))
            if len(other_combined) > 0:
                variants[f'{dataset_name}_1_3'] = concatenate_datasets([subset_1_3, other_combined])
            else:
                variants[f'{dataset_name}_1_3'] = subset_1_3
            console.print(f"  [green]{dataset_name}_1_3 创建完成: {len(variants[f'{dataset_name}_1_3']):,} 样本[/green]")
        
        # 1/2 变体：目标数据集的1/2 + 所有其他数据集
        if len(target_dataset) >= 2:
            subset_1_2 = target_dataset.select(random.sample(range(len(target_dataset)), len(target_dataset) // 2))
            if len(other_combined) > 0:
                variants[f'{dataset_name}_1_2'] = concatenate_datasets([subset_1_2, other_combined])
            else:
                variants[f'{dataset_name}_1_2'] = subset_1_2
            console.print(f"  [green]{dataset_name}_1_2 创建完成: {len(variants[f'{dataset_name}_1_2']):,} 样本[/green]")
        
        # 2x 变体：目标数据集的2倍 + 所有其他数据集
        double_dataset = concatenate_datasets([target_dataset, target_dataset])
        if len(other_combined) > 0:
            variants[f'{dataset_name}_2x'] = concatenate_datasets([double_dataset, other_combined])
        else:
            variants[f'{dataset_name}_2x'] = double_dataset
        console.print(f"  [green]{dataset_name}_2x 创建完成: {len(variants[f'{dataset_name}_2x']):,} 样本[/green]")
        
        # 3x 变体：目标数据集的3倍 + 所有其他数据集
        triple_dataset = concatenate_datasets([target_dataset, target_dataset, target_dataset])
        if len(other_combined) > 0:
            variants[f'{dataset_name}_3x'] = concatenate_datasets([triple_dataset, other_combined])
        else:
            variants[f'{dataset_name}_3x'] = triple_dataset
        console.print(f"  [green]{dataset_name}_3x 创建完成: {len(variants[f'{dataset_name}_3x']):,} 样本[/green]")
        
        return variants
    
    def build_all_datasets(self):
        """构建所有数据集"""
        console.print("[bold green]开始构建parquet数据集...[/bold green]")
        
        # 1. 加载源数据集
        source_datasets = self.load_source_datasets()
        if not source_datasets:
            console.print("[red]没有成功加载任何源数据集，退出[/red]")
            return
        
        # 2. 创建基础数据集
        base_datasets = self.create_base_datasets(source_datasets)
        
        # 3. 创建验证数据集
        validation_datasets = self.create_validation_datasets(source_datasets)
        
        # 4. 创建训练变体
        training_datasets = self.create_training_variants(base_datasets)
        
        # 5. 保存数据集
        self.save_datasets(training_datasets, validation_datasets)
        
        console.print("[bold green]parquet数据集构建完成！[/bold green]")
        console.print(f"训练数据集: {len(training_datasets)} 个")
        console.print(f"验证数据集: {len(validation_datasets)} 个")


def main():
    """主函数"""
    console.print("[bold blue]Parquet数据集构建器[/bold blue]")
    
    builder = ParquetDataBuilder()
    builder.build_all_datasets()


if __name__ == "__main__":
    main()