#!/usr/bin/env python3
"""
TikTok评论数据集构建器
用于处理/mnt/hdfs/selection/tiktok_cmt下的10个区间数据集，将doc字段转换为SFT训练格式
"""

import os
import glob
import random
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, Dataset, concatenate_datasets
from rich.console import Console
from rich.progress import Progress

# 导入基础DataBuilder类
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_builder import DataBuilder

console = Console()

class TikTokCommentDataBuilder(DataBuilder):
    """处理TikTok评论数据集的DataBuilder子类"""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        super().__init__(config_path)
        
        # TikTok评论数据集配置
        self.base_dir = "/mnt/hdfs/selection/tiktok_cmt"
        
        # 10个区间数据集
        self.interval_datasets = [
            "[0.0, 0.1)",
            "[0.1, 0.2)",
            "[0.2, 0.3)",
            "[0.3, 0.4)",
            "[0.4, 0.5)",
            "[0.5, 0.6)",
            "[0.6, 0.7)",
            "[0.7, 0.8)",
            "[0.8, 0.9)",
            "[0.9, 1.0)"
        ]
        
        console.print(f"[green]发现 {len(self.interval_datasets)} 个TikTok评论区间数据集[/green]")
        console.print(f"将生成 {1 + len(self.interval_datasets) * 4} 个训练集")
    
    def convert_doc_to_sft_format(self, doc: str, comment_id: str = None) -> List[Dict]:
        """将doc字段转换为SFT训练格式的messages"""
        if not doc or not isinstance(doc, str):
            return []
        
        # 创建简单的SFT格式：用户提供评论内容，助手回复分析
        messages = [
            {
                "role": "user",
                "content": ""
            },
            {
                "role": "assistant", 
                "content": doc
            }
        ]
        
        return messages
    
    def load_parquet_files_from_directory(self, directory_path: str) -> Dataset:
        """从目录中加载所有parquet文件并合并"""
        if not os.path.exists(directory_path):
            console.print(f"[red]目录不存在: {directory_path}[/red]")
            return None
        
        # 查找所有parquet文件
        escaped_path = glob.escape(directory_path)
        parquet_files = glob.glob(os.path.join(escaped_path, "*.parquet"))

        if not parquet_files:
            console.print(f"[yellow]目录中没有找到parquet文件: {directory_path}[/yellow]")
            return None
        
        console.print(f"  📁 找到 {len(parquet_files)} 个parquet文件")
        
        datasets_to_concat = []
        
        try:
            # 直接使用data_files加载所有parquet文件
            console.print(f"    📂 正在加载 {len(parquet_files)} 个parquet文件...")
            dataset_dict = load_dataset('parquet', data_files=parquet_files)
            combined_dataset = dataset_dict['train']
            console.print(f"    ✅ 加载成功: {len(combined_dataset):,} 总样本")
        except Exception as e:
            console.print(f"    ❌ 加载失败: {e}")
            return None

        return combined_dataset
    
    def _process_single_example(self, example: Dict) -> Dict:
        """处理单个样本，将doc字段转换为SFT格式"""
        # 获取doc字段
        doc = example.get('doc', '')
        comment_id = example.get('comment_id', '')
        
        if not doc:
            return None
        
        # 转换为SFT格式
        messages = self.convert_doc_to_sft_format(doc, comment_id)
        
        if not messages:
            return None
        
        # 转换为文本格式
        text = self.convert_messages_to_text(messages)
        
        if not text:
            return None
        
        # 创建新的example
        new_example = {
            'text': text,
            'messages': messages,
            'original_comment_id': comment_id,
            'interval': example.get('interval', ''),
            'dataset': example.get('dataset', ''),
            'category': example.get('category', ''),
        }
        
        # 保留其他有用的字段
        for key in ['language', 'language_score', 'token_num']:
            if key in example:
                new_example[key] = example[key]
        
        return new_example

    def process_tiktok_dataset(self, dataset: Dataset, dataset_name: str) -> Dataset:
        """处理TikTok数据集，将doc字段转换为SFT格式（多线程版本）"""
        console.print(f"[yellow]正在处理 {dataset_name} 数据集...[/yellow]")
        
        processed_examples = []
        total_samples = len(dataset)
        
        # 使用多线程处理
        max_workers = min(8, os.cpu_count() or 4)  # 限制最大线程数
        console.print(f"[blue]使用 {max_workers} 个线程并行处理...[/blue]")
        
        with Progress() as progress:
            task = progress.add_task(f"处理 {dataset_name}", total=total_samples)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_index = {
                    executor.submit(self._process_single_example, example): i 
                    for i, example in enumerate(dataset)
                }
                
                # 收集结果
                for future in as_completed(future_to_index):
                    try:
                        result = future.result()
                        if result is not None:
                            processed_examples.append(result)
                        
                        progress.update(task, advance=1)
                        
                        # 每1000个样本更新一次描述
                        if len(processed_examples) % 1000 == 0:
                            progress.update(task, description=f"处理 {dataset_name} (已处理: {len(processed_examples):,} 样本)")
                            
                    except Exception as e:
                        console.print(f"[red]处理样本时出错: {e}[/red]")
                        progress.update(task, advance=1)
                        continue
        
        console.print(f"[green]{dataset_name} 处理完成: {len(processed_examples):,} 样本[/green]")
        
        # 创建新的Dataset
        processed_dataset = Dataset.from_list(processed_examples)
        return processed_dataset
    
    def load_source_datasets(self) -> Dict[str, Dataset]:
        """加载TikTok评论源数据集"""
        console.print("[blue]正在加载TikTok评论数据集...[/blue]")
        
        datasets = {}
        
        for interval in self.interval_datasets:
            console.print(f"\n[cyan]处理区间: {interval}[/cyan]")
            
            # 构建训练数据路径
            train_dir = os.path.join(self.base_dir, interval, "train")
            
            # 加载训练数据
            train_dataset = self.load_parquet_files_from_directory(train_dir)
            
            if train_dataset is None:
                console.print(f"  ❌ 无法加载训练数据: {interval}")
                continue
            
            # 处理数据集，转换为SFT格式
            processed_dataset = self.process_tiktok_dataset(train_dataset, f"{interval}_train")
            
            if processed_dataset is None or len(processed_dataset) == 0:
                console.print(f"  ❌ 处理后数据集为空: {interval}")
                continue
            
            # 存储数据集
            datasets[interval] = processed_dataset
            console.print(f"  ✅ 成功处理区间 {interval}: {len(processed_dataset):,} 样本")
        
        console.print(f"\n[green]成功加载 {len(datasets)} 个区间数据集[/green]")
        return datasets
    
    def load_validation_datasets(self) -> Dict[str, Dataset]:
        """加载TikTok评论验证数据集"""
        console.print("[blue]正在加载TikTok评论验证数据集...[/blue]")
        
        validation_datasets = {}
        
        for interval in self.interval_datasets:
            console.print(f"\n[cyan]处理验证集区间: {interval}[/cyan]")
            
            # 构建验证数据路径
            val_dir = os.path.join(self.base_dir, interval, "val")
            
            # 加载验证数据
            val_dataset = self.load_parquet_files_from_directory(val_dir)
            
            if val_dataset is None:
                console.print(f"  ❌ 无法加载验证数据: {interval}")
                continue
            
            # 处理数据集，转换为SFT格式
            processed_dataset = self.process_tiktok_dataset(val_dataset, f"{interval}_val")
            
            if processed_dataset is None or len(processed_dataset) == 0:
                console.print(f"  ❌ 处理后验证数据集为空: {interval}")
                continue
            
            # 清理interval名称，去除特殊字符
            clean_interval = interval.replace("[", "").replace(")", "").replace(", ", "_").replace(".", "")
            
            # 存储验证数据集
            validation_datasets[f"{clean_interval}_val"] = processed_dataset
            console.print(f"  ✅ 成功处理验证集 {interval}: {len(processed_dataset):,} 样本")
        
        console.print(f"\n[green]成功加载 {len(validation_datasets)} 个验证数据集[/green]")
        return validation_datasets
    
    def create_training_variants(self, base_datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """创建训练变体：1个全量 + 10*4个变体扰动"""
        console.print("[blue]创建TikTok评论训练变体...[/blue]")
        
        training_datasets = {}
        dataset_names = list(base_datasets.keys())
        
        # 1. 创建全量数据集（所有10个区间数据集的组合）
        console.print("创建全量训练数据集...")
        all_datasets = list(base_datasets.values())
        if all_datasets:
            full_dataset = concatenate_datasets(all_datasets)
            training_datasets["tiktok_full_dataset"] = full_dataset
            console.print(f"[green]全量数据集创建完成: {len(full_dataset):,} 样本[/green]")
        
        # 2. 为每个区间数据集创建4种变体扰动（1/3, 1/2, 2x, 3x）+ 所有其他数据集
        console.print(f"\n开始为 {len(dataset_names)} 个区间数据集创建变体扰动...")
        
        for i, target_interval in enumerate(dataset_names, 1):
            console.print(f"\n[cyan]({i}/{len(dataset_names)}) 为 {target_interval} 创建4种变体扰动[/cyan]")
            
            target_dataset = base_datasets[target_interval]
            
            # 获取所有其他数据集
            other_datasets = [base_datasets[name] for name in dataset_names if name != target_interval]
            
            # 合并所有其他数据集
            if other_datasets:
                other_combined = concatenate_datasets(other_datasets)
            else:
                # 如果没有其他数据集，使用空数据集
                other_combined = Dataset.from_dict({})
            
            # 创建4种变体扰动
            variants = self._create_interval_variants(target_dataset, other_combined, target_interval)
            training_datasets.update(variants)
        
        console.print(f"\n[green]创建了 {len(training_datasets)} 个训练数据集[/green]")
        return training_datasets
    
    def _create_interval_variants(self, target_dataset: Dataset, other_combined: Dataset, interval_name: str) -> Dict[str, Dataset]:
        """为特定区间创建4个变体扰动"""
        variants = {}
        
        # 清理区间名称，用于文件名
        clean_name = interval_name.replace("[", "").replace(")", "").replace(", ", "_").replace(".", "")
        
        # 1/3 变体：目标数据集的1/3 + 所有其他数据集
        if len(target_dataset) >= 3:
            subset_1_3 = target_dataset.select(random.sample(range(len(target_dataset)), len(target_dataset) // 3))
            if len(other_combined) > 0:
                variants[f'tiktok_{clean_name}_1_3'] = concatenate_datasets([subset_1_3, other_combined])
            else:
                variants[f'tiktok_{clean_name}_1_3'] = subset_1_3
            console.print(f"  [green]tiktok_{clean_name}_1_3 创建完成: {len(variants[f'tiktok_{clean_name}_1_3']):,} 样本[/green]")
        
        # 1/2 变体：目标数据集的1/2 + 所有其他数据集
        if len(target_dataset) >= 2:
            subset_1_2 = target_dataset.select(random.sample(range(len(target_dataset)), len(target_dataset) // 2))
            if len(other_combined) > 0:
                variants[f'tiktok_{clean_name}_1_2'] = concatenate_datasets([subset_1_2, other_combined])
            else:
                variants[f'tiktok_{clean_name}_1_2'] = subset_1_2
            console.print(f"  [green]tiktok_{clean_name}_1_2 创建完成: {len(variants[f'tiktok_{clean_name}_1_2']):,} 样本[/green]")
        
        # 2x 变体：目标数据集的2倍 + 所有其他数据集
        double_dataset = concatenate_datasets([target_dataset, target_dataset])
        if len(other_combined) > 0:
            variants[f'tiktok_{clean_name}_2x'] = concatenate_datasets([double_dataset, other_combined])
        else:
            variants[f'tiktok_{clean_name}_2x'] = double_dataset
        console.print(f"  [green]tiktok_{clean_name}_2x 创建完成: {len(variants[f'tiktok_{clean_name}_2x']):,} 样本[/green]")
        
        # 3x 变体：目标数据集的3倍 + 所有其他数据集
        triple_dataset = concatenate_datasets([target_dataset, target_dataset, target_dataset])
        if len(other_combined) > 0:
            variants[f'tiktok_{clean_name}_3x'] = concatenate_datasets([triple_dataset, other_combined])
        else:
            variants[f'tiktok_{clean_name}_3x'] = triple_dataset
        console.print(f"  [green]tiktok_{clean_name}_3x 创建完成: {len(variants[f'tiktok_{clean_name}_3x']):,} 样本[/green]")
        
        return variants
    
    def save_datasets(self, training_datasets: Dict[str, Dataset], 
                     validation_datasets: Dict[str, Dataset]):
        """保存所有数据集到磁盘"""
        console.print("[blue]保存TikTok评论数据集到磁盘...[/blue]")
        
        # 创建保存目录
        os.makedirs("data/training", exist_ok=True)
        os.makedirs("data/validation", exist_ok=True)
        
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
        
        console.print("[bold green]所有TikTok评论数据集保存完成![/bold green]")
    
    def build_all_datasets(self):
        """构建所有TikTok评论数据集"""
        console.print("[bold green]开始构建TikTok评论数据集...[/bold green]")
        
        # 1. 加载源数据集
        source_datasets = self.load_source_datasets()
        if not source_datasets:
            console.print("[red]没有成功加载任何源数据集，退出[/red]")
            return
        
        # 2. 加载验证数据集
        validation_datasets = self.load_validation_datasets()
        
        # 3. 创建训练变体（不需要额外的token采样，因为数据已经预处理过）
        training_datasets = self.create_training_variants(source_datasets)
        
        # 4. 保存数据集
        self.save_datasets(training_datasets, validation_datasets)
        
        console.print("[bold green]TikTok评论数据集构建完成！[/bold green]")
        console.print(f"训练数据集: {len(training_datasets)} 个")
        console.print(f"验证数据集: {len(validation_datasets)} 个")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TikTok评论数据集构建器")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", 
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    console.print("[bold blue]TikTok评论数据集构建器[/bold blue]")
    console.print(f"[blue]使用配置文件: {args.config}[/blue]")
    
    builder = TikTokCommentDataBuilder(args.config)
    builder.build_all_datasets()


if __name__ == "__main__":
    main()