#!/usr/bin/env python3
"""
TikTok评论数据结构检查工具
用于分析parquet格式的TikTok评论数据，展示数据结构、统计信息和样本数据
"""

import os
import sys
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.json import JSON
from datasets import Dataset, load_dataset
import json

console = Console()

class TikTokCommentAnalyzer:
    """TikTok评论数据分析器"""
    
    def __init__(self):
        self.console = Console()
    
    def load_parquet_file(self, file_path: str) -> Tuple[Optional[Dataset], Optional[pd.DataFrame]]:
        """安全加载parquet文件，优先使用Hugging Face datasets，失败则使用pandas"""
        try:
            if not os.path.exists(file_path):
                self.console.print(f"[red]文件不存在: {file_path}[/red]")
                return None, None
            
            # 获取文件大小
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            self.console.print(f"[blue]文件大小: {file_size:.2f} MB[/blue]")
            
            # 方法1: 尝试使用Hugging Face datasets加载
            try:
                self.console.print("[yellow]尝试使用 Hugging Face datasets 加载...[/yellow]")
                dataset = load_dataset('parquet', data_files=file_path, split='train')
                self.console.print(f"[green]✓ 成功使用 Hugging Face datasets 加载数据文件[/green]")
                return dataset, None
                
            except Exception as hf_error:
                self.console.print(f"[yellow]Hugging Face datasets 加载失败: {hf_error}[/yellow]")
                self.console.print("[yellow]尝试使用 pandas 加载并转换为 Hugging Face 格式...[/yellow]")
                
                # 方法2: 使用pandas加载然后转换为Dataset
                df = pd.read_parquet(file_path)
                dataset = Dataset.from_pandas(df)
                self.console.print(f"[green]✓ 成功使用 pandas 加载并转换为 Hugging Face Dataset[/green]")
                return dataset, df
            
        except Exception as e:
            self.console.print(f"[red]所有加载方法都失败: {e}[/red]")
            return None, None
    
    def analyze_data_structure(self, dataset: Dataset, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """分析数据结构"""
        # 获取基本信息
        num_rows = len(dataset)
        columns = list(dataset.features.keys())
        
        # 如果有pandas DataFrame，使用它来获取更详细的统计信息
        if df is not None:
            analysis = {
                'shape': (num_rows, len(columns)),
                'columns': columns,
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
                'null_counts': df.isnull().sum().to_dict(),
                'unique_counts': {col: df[col].nunique() for col in df.columns},
                'hf_features': dataset.features,
                'hf_schema': str(dataset.features)
            }
        else:
            # 仅使用Dataset信息
            analysis = {
                'shape': (num_rows, len(columns)),
                'columns': columns,
                'dtypes': {col: str(dataset.features[col]) for col in columns},
                'memory_usage': None,  # 无法从Dataset直接获取
                'null_counts': {},  # 需要遍历数据才能计算，暂时跳过
                'unique_counts': {},  # 需要遍历数据才能计算，暂时跳过
                'hf_features': dataset.features,
                'hf_schema': str(dataset.features)
            }
        
        return analysis
    
    def display_basic_info(self, analysis: Dict[str, Any]):
        """显示基本信息"""
        # 创建基本信息表格
        info_table = Table(title="数据集基本信息", show_header=True, header_style="bold magenta")
        info_table.add_column("属性", style="cyan")
        info_table.add_column("值", style="green")
        
        info_table.add_row("数据行数", f"{analysis['shape'][0]:,}")
        info_table.add_row("数据列数", f"{analysis['shape'][1]:,}")
        
        if analysis['memory_usage'] is not None:
            info_table.add_row("内存使用", f"{analysis['memory_usage']:.2f} MB")
        else:
            info_table.add_row("内存使用", "N/A (仅Dataset)")
        
        self.console.print(info_table)
    
    def display_hf_schema(self, analysis: Dict[str, Any]):
        """显示Hugging Face schema信息"""
        self.console.print(f"\n[bold blue]Hugging Face Schema:[/bold blue]")
        
        # 创建schema表格
        schema_table = Table(title="Hugging Face Features Schema", show_header=True, header_style="bold magenta")
        schema_table.add_column("列名", style="cyan")
        schema_table.add_column("Hugging Face 类型", style="yellow")
        
        for col_name, feature in analysis['hf_features'].items():
            schema_table.add_row(col_name, str(feature))
        
        self.console.print(schema_table)
    
    def display_first_complete_record(self, dataset: Dataset):
        """显示第一条完整数据记录"""
        if len(dataset) == 0:
            self.console.print("[red]数据集为空，无法显示第一条记录[/red]")
            return
        
        self.console.print(f"\n[bold blue]第一条完整数据记录:[/bold blue]")
        
        first_record = dataset[0]
        
        # 使用Rich的JSON显示功能来美化输出
        self.console.print(Panel(
            JSON.from_data(first_record, indent=2),
            title="[bold green]完整数据记录 #1[/bold green]",
            border_style="green"
        ))
    
    def display_column_info(self, analysis: Dict[str, Any]):
        """显示列信息"""
        # 创建列信息表格
        col_table = Table(title="列信息详情", show_header=True, header_style="bold magenta")
        col_table.add_column("列名", style="cyan")
        col_table.add_column("数据类型", style="yellow")
        col_table.add_column("空值数量", style="red")
        col_table.add_column("唯一值数量", style="green")
        
        for col in analysis['columns']:
            # 处理可能为空的统计信息
            null_count = analysis['null_counts'].get(col, "N/A")
            unique_count = analysis['unique_counts'].get(col, "N/A")
            
            null_count_str = f"{null_count:,}" if isinstance(null_count, (int, float)) else str(null_count)
            unique_count_str = f"{unique_count:,}" if isinstance(unique_count, (int, float)) else str(unique_count)
            
            col_table.add_row(
                col,
                str(analysis['dtypes'][col]),
                null_count_str,
                unique_count_str
            )
        
        self.console.print(col_table)
    
    def display_sample_data(self, dataset: Dataset, n_samples: int = 5):
        """显示样本数据"""
        self.console.print(f"\n[bold blue]前 {n_samples} 行数据样本:[/bold blue]")
        
        # 创建样本数据表格
        sample_table = Table(show_header=True, header_style="bold magenta")
        
        # 添加列
        columns = list(dataset.features.keys())
        for col in columns:
            sample_table.add_column(col, style="cyan", max_width=30)
        
        # 添加数据行
        for idx in range(min(n_samples, len(dataset))):
            row_data = []
            record = dataset[idx]
            for col in columns:
                value = str(record[col])
                # 截断过长的文本
                if len(value) > 50:
                    value = value[:47] + "..."
                row_data.append(value)
            sample_table.add_row(*row_data)
        
        self.console.print(sample_table)
    
    def display_text_statistics(self, dataset: Dataset, df: Optional[pd.DataFrame] = None):
        """显示文本统计信息（如果有文本列）"""
        text_columns = []
        
        # 如果有pandas DataFrame，使用它来分析文本列
        if df is not None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    # 检查是否为文本列
                    sample_values = df[col].dropna().head(10)
                    if any(isinstance(val, str) and len(val) > 10 for val in sample_values):
                        text_columns.append(col)
        else:
            # 仅使用Dataset，检查前几条记录来判断文本列
            if len(dataset) > 0:
                sample_records = dataset[:min(10, len(dataset))]
                for col in dataset.features.keys():
                    # 检查是否为字符串类型且长度较长
                    sample_values = [sample_records[col][i] for i in range(len(sample_records[col]))]
                    if any(isinstance(val, str) and len(val) > 10 for val in sample_values if val is not None):
                        text_columns.append(col)
        
        if text_columns:
            self.console.print(f"\n[bold blue]文本列统计信息:[/bold blue]")
            
            text_table = Table(show_header=True, header_style="bold magenta")
            text_table.add_column("列名", style="cyan")
            text_table.add_column("平均长度", style="green")
            text_table.add_column("最大长度", style="yellow")
            text_table.add_column("最小长度", style="red")
            
            for col in text_columns:
                if df is not None:
                    # 使用pandas计算统计信息
                    text_data = df[col].dropna().astype(str)
                    lengths = text_data.str.len()
                    
                    text_table.add_row(
                        col,
                        f"{lengths.mean():.1f}",
                        f"{lengths.max():,}",
                        f"{lengths.min():,}"
                    )
                else:
                    # 使用Dataset计算统计信息（较慢，但可行）
                    lengths = []
                    for i in range(len(dataset)):
                        val = dataset[i][col]
                        if val is not None and isinstance(val, str):
                            lengths.append(len(val))
                    
                    if lengths:
                        import statistics
                        text_table.add_row(
                            col,
                            f"{statistics.mean(lengths):.1f}",
                            f"{max(lengths):,}",
                            f"{min(lengths):,}"
                        )
                    else:
                        text_table.add_row(col, "N/A", "N/A", "N/A")
            
            self.console.print(text_table)
    
    def analyze_file(self, file_path: str, n_samples: int = 5):
        """分析单个文件"""
        self.console.print(Panel(f"[bold green]分析文件: {file_path}[/bold green]"))
        
        # 加载数据
        dataset, df = self.load_parquet_file(file_path)
        if dataset is None:
            return
        
        # 分析数据结构
        analysis = self.analyze_data_structure(dataset, df)
        
        # 显示各种信息
        self.display_basic_info(analysis)
        self.display_hf_schema(analysis)
        self.display_first_complete_record(dataset)
        self.display_column_info(analysis)
        self.display_sample_data(dataset, n_samples)
        self.display_text_statistics(dataset, df)
        
        self.console.print(f"\n[bold green]✓ 文件分析完成[/bold green]")
    
    def analyze_directory(self, dir_path: str, n_samples: int = 5):
        """分析目录中的所有parquet文件"""
        parquet_files = list(Path(dir_path).glob("*.parquet"))
        
        if not parquet_files:
            self.console.print(f"[red]在目录 {dir_path} 中未找到parquet文件[/red]")
            return
        
        self.console.print(f"[blue]找到 {len(parquet_files)} 个parquet文件[/blue]")
        
        for i, file_path in enumerate(parquet_files, 1):
            self.console.print(f"\n[bold cyan]({i}/{len(parquet_files)}) 分析文件: {file_path.name}[/bold cyan]")
            self.analyze_file(str(file_path), n_samples)
            
            if i < len(parquet_files):
                self.console.print("\n" + "="*80 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TikTok评论数据结构检查工具")
    parser.add_argument("--path",
                        default='/mnt/hdfs/selection/tiktok_cmt/[0.0, 0.1)/train/part-00052-601b619f-22bc-43d1-87b4-9c483584fc06-c000.snappy.parquet',
                        help="parquet文件路径或包含parquet文件的目录路径")
    parser.add_argument("--samples", "-s", type=int, default=5, help="显示的样本数据行数 (默认: 5)")
    
    args = parser.parse_args()
    
    analyzer = TikTokCommentAnalyzer()
    
    if os.path.isfile(args.path):
        # 分析单个文件
        analyzer.analyze_file(args.path, args.samples)
    elif os.path.isdir(args.path):
        # 分析目录中的所有文件
        analyzer.analyze_directory(args.path, args.samples)
    else:
        console.print(f"[red]路径不存在: {args.path}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()