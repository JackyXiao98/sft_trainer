#!/usr/bin/env python3
"""
评估脚本
计算训练好的模型在验证集上的样本级别平均loss
"""

import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from rich.console import Console
from rich.progress import Progress
import json
import csv
from typing import Dict, List

console = Console()

def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # 初始化分布式进程组
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

class ModelEvaluator:
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """初始化评估器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化分布式环境
        self.is_distributed, self.rank, self.world_size, self.local_rank = setup_distributed()
        
        if self.is_distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
            if self.rank == 0:
                console.print(f"[green]分布式评估器初始化完成[/green]")
                console.print(f"World Size: {self.world_size}, Rank: {self.rank}")
                console.print(f"使用设备: {self.device}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            console.print(f"[green]单GPU评估器初始化完成[/green]")
            console.print(f"使用设备: {self.device}")
    
    def load_model_and_tokenizer(self, model_path: str):
        """加载训练好的模型和tokenizer"""
        if not self.is_distributed or self.rank == 0:
            console.print(f"[blue]加载模型: {model_path}[/blue]")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config['model']['trust_remote_code']
        )
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        if self.is_distributed:
            # 分布式模式：不使用device_map，手动设置设备
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=self.config['model']['trust_remote_code'],
                dtype=torch.bfloat16 if self.config['misc']['bf16'] else torch.float16,
            )
            self.model = self.model.to(self.device)
            # 使用DistributedDataParallel包装模型
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            # 单GPU模式：使用device_map="auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=self.config['model']['trust_remote_code'],
                dtype=torch.bfloat16 if self.config['misc']['bf16'] else torch.float16,
                device_map="auto"
            )
        
        self.model.eval()
        
        if not self.is_distributed or self.rank == 0:
            console.print(f"[green]模型加载完成[/green]")
            # 获取模型参数量（处理DDP包装的情况）
            model_for_params = self.model.module if hasattr(self.model, 'module') else self.model
            console.print(f"模型参数量: {model_for_params.num_parameters():,}")
            if self.is_distributed:
                console.print(f"[yellow]使用分布式数据并行 (DDP)[/yellow]")
    
    def load_validation_dataset(self, dataset_path: str):
        """加载验证数据集"""
        console.print(f"[blue]加载验证数据集: {dataset_path}[/blue]")
        
        dataset = load_from_disk(dataset_path)
        console.print(f"[green]验证数据集加载完成: {len(dataset):,} 样本[/green]")
        
        return dataset
    
    def tokenize_function(self, examples):
        """tokenize函数"""
        # 获取评估时的最大序列长度，如果没有设置则使用训练时的长度
        eval_max_length = self.config['model'].get('max_seq_length_eval', 
                                                   self.config['model']['max_seq_length'])
        
        # 对文本进行tokenize
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=eval_max_length,
            return_tensors=None
        )
        
        # 设置labels为input_ids的副本（用于计算loss）
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def compute_sample_loss(self, batch):
        """计算单个batch的样本级别loss"""
        try:
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with torch.no_grad():
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            # 获取logits和计算loss
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算每个样本的loss
            batch_size, seq_len, vocab_size = shift_logits.shape
            
            # 重塑为 (batch_size * seq_len, vocab_size)
            flat_logits = shift_logits.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)
            
            # 计算每个token的loss
            token_losses = F.cross_entropy(
                flat_logits, 
                flat_labels, 
                reduction='none',
                ignore_index=-100
            )
            
            # 重塑回 (batch_size, seq_len)
            token_losses = token_losses.view(batch_size, seq_len)
            
            # 计算每个样本的平均loss（忽略padding）
            sample_losses = []
            for i in range(batch_size):
                # 获取有效token的mask（非padding且非-100）
                valid_mask = (shift_labels[i] != -100) & (shift_labels[i] != self.tokenizer.pad_token_id)
                if valid_mask.sum() > 0:
                    sample_loss = token_losses[i][valid_mask].mean()
                    sample_losses.append(sample_loss.item())
                else:
                    # 如果没有有效token，跳过这个样本
                    continue
            
            return sample_losses
        
        except RuntimeError as e:
            if "CUDA" in str(e):
                console.print(f"[red]CUDA错误: {e}[/red]")
                console.print("[yellow]尝试清理CUDA缓存并重试...[/yellow]")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 返回空列表，跳过这个batch
                return []
            else:
                raise e
        except Exception as e:
            console.print(f"[red]计算loss时发生错误: {e}[/red]")
            return []
    
    def evaluate_single_dataset(self, dataset_path: str) -> float:
        """在单个数据集上评估模型（假设模型已加载）"""
        # 加载数据集
        dataset = self.load_validation_dataset(dataset_path)
        
        # tokenize数据集
        if not self.is_distributed or self.rank == 0:
            console.print("[blue]对数据集进行tokenize...[/blue]")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=1, #使用多线程有bug
            desc="Tokenizing dataset"
        )
        
        # 创建自定义数据整理器来处理已tokenized的数据
        def data_collator(features):
            # 提取已tokenized的字段
            input_ids = [torch.tensor(feature['input_ids']) for feature in features]
            attention_mask = [torch.tensor(feature['attention_mask']) for feature in features]
            labels = [torch.tensor(feature['labels']) for feature in features]
            
            # 进行padding
            from torch.nn.utils.rnn import pad_sequence
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
            labels = pad_sequence(labels, batch_first=True, padding_value=-100)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        
        # 使用配置中的批次大小
        eval_batch_size = self.config['training']['per_device_eval_batch_size']
        if not self.is_distributed or self.rank == 0:
            console.print(f"[yellow]使用评估批次大小: {eval_batch_size}[/yellow]")
        
        # 创建采样器
        if self.is_distributed:
            sampler = DistributedSampler(
                tokenized_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=False
            )
            dataloader = DataLoader(
                tokenized_dataset,
                batch_size=eval_batch_size,
                sampler=sampler,
                collate_fn=data_collator
            )
        else:
            dataloader = DataLoader(
                tokenized_dataset,
                batch_size=eval_batch_size,
                collate_fn=data_collator,
                shuffle=False
            )
        
        # 计算所有样本的loss
        if not self.is_distributed or self.rank == 0:
            console.print("[blue]计算样本级别loss...[/blue]")
        all_sample_losses = []
        
        # 只在rank 0显示进度条
        if not self.is_distributed or self.rank == 0:
            with Progress() as progress:
                task = progress.add_task("评估进度", total=len(dataloader))
                
                for batch in dataloader:
                    sample_losses = self.compute_sample_loss(batch)
                    all_sample_losses.extend(sample_losses)
                    progress.advance(task)
        else:
            for batch in dataloader:
                sample_losses = self.compute_sample_loss(batch)
                all_sample_losses.extend(sample_losses)
        
        # 分布式环境下聚合结果
        if self.is_distributed:
            # 收集所有进程的样本损失
            all_losses_tensor = torch.tensor(all_sample_losses, device=self.device)
            
            # 收集所有进程的损失数量
            num_samples = torch.tensor([len(all_sample_losses)], device=self.device)
            gathered_nums = [torch.zeros_like(num_samples) for _ in range(self.world_size)]
            dist.all_gather(gathered_nums, num_samples)
            
            # 计算最大样本数，用于padding
            max_samples = max(gathered_nums).item()
            
            # Padding损失张量到相同长度
            if len(all_sample_losses) < max_samples:
                padding = torch.zeros(max_samples - len(all_sample_losses), device=self.device)
                all_losses_tensor = torch.cat([all_losses_tensor, padding])
            
            # 收集所有进程的损失
            gathered_losses = [torch.zeros(max_samples, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(gathered_losses, all_losses_tensor)
            
            # 在rank 0上计算最终结果
            if self.rank == 0:
                final_losses = []
                for i, (losses, num) in enumerate(zip(gathered_losses, gathered_nums)):
                    final_losses.extend(losses[:num.item()].cpu().tolist())
                
                if len(final_losses) > 0:
                    avg_loss = sum(final_losses) / len(final_losses)
                else:
                    avg_loss = float('inf')
                
                console.print(f"[green]分布式评估完成[/green]")
                console.print(f"总样本数量: {len(final_losses):,}")
                console.print(f"平均loss: {avg_loss:.6f}")
                
                return avg_loss
            else:
                return 0.0  # 非主进程返回0
        else:
            # 单GPU模式
            if len(all_sample_losses) > 0:
                avg_loss = sum(all_sample_losses) / len(all_sample_losses)
            else:
                avg_loss = float('inf')
            
            console.print(f"[green]评估完成[/green]")
            console.print(f"样本数量: {len(all_sample_losses):,}")
            console.print(f"平均loss: {avg_loss:.6f}")
            
            return avg_loss

    def evaluate_on_dataset(self, model_path: str, dataset_path: str) -> float:
        """在指定数据集上评估模型（兼容性方法）"""
        # 加载模型
        self.load_model_and_tokenizer(model_path)
        
        # 评估单个数据集
        return self.evaluate_single_dataset(dataset_path)
    
    def evaluate_on_multiple_datasets(self, model_path: str, dataset_paths: List[str], 
                                    train_dataset_name: str, validation_dataset_names: List[str],
                                    output_file: str = "results.csv") -> Dict[str, float]:
        """在多个数据集上评估模型（只加载一次模型）"""
        if not self.is_distributed or self.rank == 0:
            console.print(f"[bold blue]开始批量评估[/bold blue]")
            console.print(f"模型: {model_path}")
            console.print(f"验证集数量: {len(dataset_paths)}")
        
        # 只加载一次模型
        self.load_model_and_tokenizer(model_path)
        
        results = {}
        
        # 逐个评估验证集
        for i, (dataset_path, val_name) in enumerate(zip(dataset_paths, validation_dataset_names)):
            if not self.is_distributed or self.rank == 0:
                console.print(f"[bold cyan]评估进度: {i+1}/{len(dataset_paths)}[/bold cyan]")
                console.print(f"当前验证集: {val_name}")
            
            try:
                avg_loss = self.evaluate_single_dataset(dataset_path)
                results[val_name] = avg_loss
                
                # 只在主进程保存结果
                if not self.is_distributed or self.rank == 0:
                    # 保存单个结果
                    result = {
                        'train_dataset_name': train_dataset_name,
                        'validation_dataset_name': val_name,
                        'loss': avg_loss
                    }
                    self.save_result(result, output_file)
                    
                    console.print(f"[green]✓ {val_name}: {avg_loss:.6f}[/green]")
                
            except Exception as e:
                if not self.is_distributed or self.rank == 0:
                    console.print(f"[red]✗ {val_name}: 评估失败 - {e}[/red]")
                results[val_name] = float('inf')
                
                # 只在主进程保存失败结果
                if not self.is_distributed or self.rank == 0:
                    result = {
                        'train_dataset_name': train_dataset_name,
                        'validation_dataset_name': val_name,
                        'loss': float('inf')
                    }
                    self.save_result(result, output_file)
        
        if not self.is_distributed or self.rank == 0:
            console.print(f"[bold green]批量评估完成![/bold green]")
            console.print("结果摘要:")
            for val_name, loss in results.items():
                if loss == float('inf'):
                    console.print(f"  {val_name}: [red]失败[/red]")
                else:
                    console.print(f"  {val_name}: [green]{loss:.6f}[/green]")
        
        return results
    
    def save_result(self, result: Dict, output_file: str = "results.csv"):
        """保存评估结果"""
        # 检查文件是否存在
        file_exists = os.path.exists(output_file)
        
        # 写入CSV文件
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['train_dataset_name', 'validation_dataset_name', 'loss']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writeheader()
            
            # 写入结果
            writer.writerow(result)
        
        console.print(f"[green]结果已保存到: {output_file}[/green]")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--train_dataset_name", type=str, required=True, help="训练数据集名称")
    parser.add_argument("--output_file", type=str, default="results.csv", help="结果输出文件")
    parser.add_argument("--config", type=str, default="configs/training_config_thoth.yaml", help="配置文件路径")
    
    # 支持单个验证集（向后兼容）
    parser.add_argument("--dataset_path", type=str, help="单个验证数据集路径")
    parser.add_argument("--validation_dataset_name", type=str, help="单个验证数据集名称")
    
    # 支持多个验证集（新功能）
    parser.add_argument("--dataset_paths", type=str, nargs='+', help="多个验证数据集路径")
    parser.add_argument("--validation_dataset_names", type=str, nargs='+', help="多个验证数据集名称")
    
    args = parser.parse_args()
    
    try:
        # 初始化评估器
        evaluator = ModelEvaluator(args.config)
        
        # 检查参数组合
        if args.dataset_paths and args.validation_dataset_names:
            # 批量评估模式
            if len(args.dataset_paths) != len(args.validation_dataset_names):
                console.print("[red]错误: dataset_paths和validation_dataset_names的数量必须相同[/red]")
                return
            
            # 只在主进程显示信息
            is_distributed, rank, _, _ = setup_distributed() if not hasattr(evaluator, 'is_distributed') else (evaluator.is_distributed, evaluator.rank, evaluator.world_size, evaluator.local_rank)
            
            if not is_distributed or rank == 0:
                console.print(f"[bold blue]开始批量评估[/bold blue]")
                console.print(f"模型: {args.model_path}")
                console.print(f"验证集数量: {len(args.dataset_paths)}")
            
            # 执行批量评估
            results = evaluator.evaluate_on_multiple_datasets(
                model_path=args.model_path,
                dataset_paths=args.dataset_paths,
                train_dataset_name=args.train_dataset_name,
                validation_dataset_names=args.validation_dataset_names,
                output_file=args.output_file
            )
            
            if not is_distributed or rank == 0:
                console.print(f"[bold green]批量评估完成![/bold green]")
                console.print(f"训练数据集: {args.train_dataset_name}")
                console.print("所有验证集结果:")
                for val_name, loss in results.items():
                    if loss == float('inf'):
                        console.print(f"  {val_name}: [red]失败[/red]")
                    else:
                        console.print(f"  {val_name}: [green]{loss:.6f}[/green]")
        
        elif args.dataset_path and args.validation_dataset_name:
            # 单个评估模式（向后兼容）
            is_distributed, rank, _, _ = setup_distributed() if not hasattr(evaluator, 'is_distributed') else (evaluator.is_distributed, evaluator.rank, evaluator.world_size, evaluator.local_rank)
            
            if not is_distributed or rank == 0:
                console.print(f"[bold blue]开始评估[/bold blue]")
                console.print(f"模型: {args.model_path}")
                console.print(f"验证集: {args.dataset_path}")
            
            avg_loss = evaluator.evaluate_on_dataset(args.model_path, args.dataset_path)
            
            if not is_distributed or rank == 0:
                # 保存结果
                result = {
                    'train_dataset_name': args.train_dataset_name,
                    'validation_dataset_name': args.validation_dataset_name,
                    'loss': avg_loss
                }
                
                evaluator.save_result(result, args.output_file)
                
                console.print(f"[bold green]评估完成![/bold green]")
                console.print(f"训练数据集: {args.train_dataset_name}")
                console.print(f"验证数据集: {args.validation_dataset_name}")
                console.print(f"平均loss: {avg_loss:.6f}")
        
        else:
            console.print("[red]错误: 请提供以下参数组合之一:[/red]")
            console.print("  1. 单个评估: --dataset_path 和 --validation_dataset_name")
            console.print("  2. 批量评估: --dataset_paths 和 --validation_dataset_names")
    
    except Exception as e:
        console.print(f"[red]评估过程中发生错误: {e}[/red]")
        raise
    finally:
        # 清理分布式环境
        cleanup_distributed()

if __name__ == "__main__":
    main()