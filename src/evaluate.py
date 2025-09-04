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
from rich.console import Console
from rich.progress import Progress
import json
import csv
from typing import Dict, List

console = Console()

class ModelEvaluator:
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """初始化评估器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[green]评估器初始化完成[/green]")
        console.print(f"使用设备: {self.device}")
    
    def load_model_and_tokenizer(self, model_path: str):
        """加载训练好的模型和tokenizer"""
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=self.config['model']['trust_remote_code'],
            torch_dtype=torch.bfloat16 if self.config['misc']['bf16'] else torch.float16,
            device_map="auto"
        )
        
        self.model.eval()
        console.print(f"[green]模型加载完成[/green]")
        console.print(f"模型参数量: {self.model.num_parameters():,}")
    
    def load_validation_dataset(self, dataset_path: str):
        """加载验证数据集"""
        console.print(f"[blue]加载验证数据集: {dataset_path}[/blue]")
        
        dataset = load_from_disk(dataset_path)
        console.print(f"[green]验证数据集加载完成: {len(dataset):,} 样本[/green]")
        
        return dataset
    
    def tokenize_function(self, examples):
        """tokenize函数"""
        # 对文本进行tokenize
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=self.config['model']['max_seq_length'],
            return_tensors=None
        )
        
        # 设置labels为input_ids的副本（用于计算loss）
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def compute_sample_loss(self, batch):
        """计算单个batch的样本级别loss"""
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
    
    def evaluate_on_dataset(self, model_path: str, dataset_path: str) -> float:
        """在指定数据集上评估模型"""
        # 加载模型
        self.load_model_and_tokenizer(model_path)
        
        # 加载数据集
        dataset = self.load_validation_dataset(dataset_path)
        
        # tokenize数据集
        console.print("[blue]对数据集进行tokenize...[/blue]")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 创建数据加载器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.config['training']['per_device_eval_batch_size'],
            collate_fn=data_collator,
            shuffle=False
        )
        
        # 计算所有样本的loss
        console.print("[blue]计算样本级别loss...[/blue]")
        all_sample_losses = []
        
        with Progress() as progress:
            task = progress.add_task("评估进度", total=len(dataloader))
            
            for batch in dataloader:
                sample_losses = self.compute_sample_loss(batch)
                all_sample_losses.extend(sample_losses)
                progress.advance(task)
        
        # 计算平均loss
        if len(all_sample_losses) > 0:
            avg_loss = sum(all_sample_losses) / len(all_sample_losses)
        else:
            avg_loss = float('inf')
        
        console.print(f"[green]评估完成[/green]")
        console.print(f"样本数量: {len(all_sample_losses):,}")
        console.print(f"平均loss: {avg_loss:.6f}")
        
        return avg_loss
    
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
    parser.add_argument("--dataset_path", type=str, required=True, help="验证数据集路径")
    parser.add_argument("--train_dataset_name", type=str, required=True, help="训练数据集名称")
    parser.add_argument("--validation_dataset_name", type=str, required=True, help="验证数据集名称")
    parser.add_argument("--output_file", type=str, default="results.csv", help="结果输出文件")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = ModelEvaluator(args.config)
    
    # 执行评估
    console.print(f"[bold blue]开始评估[/bold blue]")
    console.print(f"模型: {args.model_path}")
    console.print(f"验证集: {args.dataset_path}")
    
    avg_loss = evaluator.evaluate_on_dataset(args.model_path, args.dataset_path)
    
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

if __name__ == "__main__":
    main()