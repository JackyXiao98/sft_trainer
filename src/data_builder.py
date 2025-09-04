#!/usr/bin/env python3
"""
数据合成器模块
用于从mixture-of-thoughts数据集生成13个训练数据集和3个验证数据集
"""

import os
import yaml
import random
from typing import List, Dict, Tuple
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from rich.console import Console
from rich.progress import Progress, TaskID

console = Console()

class DataBuilder:
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """初始化数据构建器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['model_name'],
            trust_remote_code=self.config['model']['trust_remote_code']
        )
        
        # 设置随机种子
        random.seed(self.config['training']['seed'])
        
        # 数据配置
        self.train_token_limit = self.config['data']['token_limits']['train_base']
        self.val_token_limit = self.config['data']['token_limits']['validation']
        
        console.print(f"[green]数据构建器初始化完成[/green]")
        console.print(f"模型: {self.config['model']['model_name']}")
        console.print(f"训练集token限制: {self.train_token_limit:,}")
        console.print(f"验证集token限制: {self.val_token_limit:,}")
    
    def load_source_datasets(self) -> Dict[str, Dataset]:
        """加载源数据集"""
        console.print("[blue]正在加载mixture-of-thoughts数据集...[/blue]")
        
        datasets = {}
        for subset in self.config['data']['dataset_config_names']:
            console.print(f"加载 {subset} 子集...")
            dataset = load_dataset(
                self.config['data']['dataset_name'],
                subset,
                split='train'
            )
            datasets[subset] = dataset
            console.print(f"{subset} 子集加载完成，共 {len(dataset):,} 个样本")
        
        return datasets
    
    def sample_by_token_count(self, dataset: Dataset, token_limit: int, subset_name: str) -> Dataset:
        """根据token数量采样数据集"""
        console.print(f"[yellow]正在从 {subset_name} 采样 {token_limit:,} tokens...[/yellow]")
        
        sampled_examples = []
        total_tokens = 0
        
        with Progress() as progress:
            task = progress.add_task(f"采样 {subset_name}", total=len(dataset))
            
            for i, example in enumerate(dataset):
                # 对文本进行tokenize
                text = example.get('text', '')
                if not text:
                    continue
                
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                token_count = len(tokens)
                
                # 检查是否超过限制
                if total_tokens + token_count > token_limit:
                    break
                
                sampled_examples.append(example)
                total_tokens += token_count
                
                progress.update(task, advance=1)
                
                if i % 1000 == 0:
                    progress.update(task, description=f"采样 {subset_name} (已采样: {total_tokens:,} tokens)")
        
        console.print(f"[green]{subset_name} 采样完成: {len(sampled_examples):,} 样本, {total_tokens:,} tokens[/green]")
        
        # 创建新的Dataset
        sampled_dataset = Dataset.from_list(sampled_examples)
        return sampled_dataset
    
    def create_base_datasets(self, source_datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """创建基础数据集 D1, D2, D3"""
        console.print("[blue]创建基础数据集...[/blue]")
        
        base_datasets = {}
        for subset_name in ['math', 'code', 'science']:
            dataset_name = f"D{['math', 'code', 'science'].index(subset_name) + 1}"
            base_datasets[dataset_name] = self.sample_by_token_count(
                source_datasets[subset_name],
                self.train_token_limit,
                f"{dataset_name} ({subset_name})"
            )
        
        return base_datasets
    
    def create_validation_datasets(self, source_datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """创建验证数据集 V1, V2, V3"""
        console.print("[blue]创建验证数据集...[/blue]")
        
        validation_datasets = {}
        for subset_name in ['math', 'code', 'science']:
            dataset_name = f"V{['math', 'code', 'science'].index(subset_name) + 1}"
            validation_datasets[dataset_name] = self.sample_by_token_count(
                source_datasets[subset_name],
                self.val_token_limit,
                f"{dataset_name} ({subset_name})"
            )
        
        return validation_datasets
    
    def create_training_variants(self, base_datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """创建13个训练变体"""
        console.print("[blue]创建训练数据集变体...[/blue]")
        
        training_datasets = {}
        
        # 1. 基线数据集
        baseline = concatenate_datasets([
            base_datasets['D1'],
            base_datasets['D2'],
            base_datasets['D3']
        ])
        training_datasets['baseline'] = baseline
        console.print(f"[green]基线数据集创建完成: {len(baseline):,} 样本[/green]")
        
        # 2. 扰动 D1 (math)
        d1_variants = self._create_domain_variants(base_datasets, 'D1', 'D2', 'D3', 'math')
        training_datasets.update(d1_variants)
        
        # 3. 扰动 D2 (code)
        d2_variants = self._create_domain_variants(base_datasets, 'D2', 'D1', 'D3', 'code')
        training_datasets.update(d2_variants)
        
        # 4. 扰动 D3 (science)
        d3_variants = self._create_domain_variants(base_datasets, 'D3', 'D1', 'D2', 'science')
        training_datasets.update(d3_variants)
        
        return training_datasets
    
    def _create_domain_variants(self, base_datasets: Dict[str, Dataset], 
                               target_key: str, other_key1: str, other_key2: str, 
                               domain_name: str) -> Dict[str, Dataset]:
        """为特定领域创建4个变体"""
        variants = {}
        target_dataset = base_datasets[target_key]
        other_dataset1 = base_datasets[other_key1]
        other_dataset2 = base_datasets[other_key2]
        
        # 1/3 变体
        subset_1_3 = target_dataset.select(random.sample(range(len(target_dataset)), len(target_dataset) // 3))
        variants[f'{domain_name}_1_3'] = concatenate_datasets([subset_1_3, other_dataset1, other_dataset2])
        
        # 1/2 变体
        subset_1_2 = target_dataset.select(random.sample(range(len(target_dataset)), len(target_dataset) // 2))
        variants[f'{domain_name}_1_2'] = concatenate_datasets([subset_1_2, other_dataset1, other_dataset2])
        
        # 2x 变体
        double_dataset = concatenate_datasets([target_dataset, target_dataset])
        variants[f'{domain_name}_2x'] = concatenate_datasets([double_dataset, other_dataset1, other_dataset2])
        
        # 3x 变体
        triple_dataset = concatenate_datasets([target_dataset, target_dataset, target_dataset])
        variants[f'{domain_name}_3x'] = concatenate_datasets([triple_dataset, other_dataset1, other_dataset2])
        
        for variant_name, dataset in variants.items():
            console.print(f"[green]{variant_name} 创建完成: {len(dataset):,} 样本[/green]")
        
        return variants
    
    def save_datasets(self, training_datasets: Dict[str, Dataset], 
                     validation_datasets: Dict[str, Dataset]):
        """保存所有数据集到磁盘"""
        console.print("[blue]保存数据集到磁盘...[/blue]")
        
        # 保存训练数据集
        for name, dataset in training_datasets.items():
            save_path = f"data/training/{name}"
            dataset.save_to_disk(save_path)
            console.print(f"[green]训练数据集 {name} 已保存到 {save_path}[/green]")
        
        # 保存验证数据集
        for name, dataset in validation_datasets.items():
            save_path = f"data/validation/{name}"
            dataset.save_to_disk(save_path)
            console.print(f"[green]验证数据集 {name} 已保存到 {save_path}[/green]")
        
        console.print("[bold green]所有数据集保存完成![/bold green]")
    
    def build_all_datasets(self):
        """构建所有数据集的主函数"""
        console.print("[bold blue]开始构建SFT Scaling Law数据集...[/bold blue]")
        
        # 1. 加载源数据集
        source_datasets = self.load_source_datasets()
        
        # 2. 创建基础数据集
        base_datasets = self.create_base_datasets(source_datasets)
        
        # 3. 创建验证数据集
        validation_datasets = self.create_validation_datasets(source_datasets)
        
        # 4. 创建训练变体
        training_datasets = self.create_training_variants(base_datasets)
        
        # 5. 保存所有数据集
        self.save_datasets(training_datasets, validation_datasets)
        
        console.print("[bold green]数据集构建完成![/bold green]")
        console.print(f"训练数据集: {len(training_datasets)} 个")
        console.print(f"验证数据集: {len(validation_datasets)} 个")
        console.print(f"总计: {len(training_datasets) * len(validation_datasets)} 个实验组合")

def main():
    """主函数"""
    builder = DataBuilder()
    builder.build_all_datasets()

if __name__ == "__main__":
    main()