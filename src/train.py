#!/usr/bin/env python3
"""
SFT训练脚本
使用TRL的SFTTrainer进行监督微调，支持FSDP和Flash Attention 2
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import yaml
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer
import wandb
from rich.console import Console

console = Console()

class SFTTrainingPipeline:
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """初始化训练管道"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        console.print(f"[green]训练管道初始化完成[/green]")
        console.print(f"模型: {self.config['model']['model_name']}")
    
    def setup_model_and_tokenizer(self):
        """设置模型和tokenizer"""
        console.print("[blue]加载模型和tokenizer...[/blue]")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['model_name'],
            trust_remote_code=self.config['model']['trust_remote_code']
        )
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            console.print(f"[yellow]设置 pad_token 为 eos_token: {self.tokenizer.eos_token}[/yellow]")
        
        # 加载模型
        model_kwargs = {
            'trust_remote_code': self.config['model']['trust_remote_code'],
            'dtype': torch.bfloat16 if self.config['misc']['bf16'] else torch.float16,
            'device_map': None,  # FSDP会处理设备分配
        }
        
        if self.config['model']['use_flash_attention_2']:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['model_name'],
            **model_kwargs
        )
        
        # 确保模型配置与分词器对齐
        if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id != self.tokenizer.pad_token_id:
            console.print(f"[yellow]对齐模型 pad_token_id: {self.model.config.pad_token_id} -> {self.tokenizer.pad_token_id}[/yellow]")
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        if hasattr(self.model.config, 'bos_token_id') and self.model.config.bos_token_id != self.tokenizer.bos_token_id:
            console.print(f"[yellow]对齐模型 bos_token_id: {self.model.config.bos_token_id} -> {self.tokenizer.bos_token_id}[/yellow]")
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
        
        if hasattr(self.model.config, 'eos_token_id') and self.model.config.eos_token_id != self.tokenizer.eos_token_id:
            console.print(f"[yellow]对齐模型 eos_token_id: {self.model.config.eos_token_id} -> {self.tokenizer.eos_token_id}[/yellow]")
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        
        # 调整模型的embedding层大小以匹配分词器
        if len(self.tokenizer) > self.model.config.vocab_size:
            console.print(f"[yellow]调整模型词汇表大小: {self.model.config.vocab_size} -> {len(self.tokenizer)}[/yellow]")
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 启用梯度检查点并禁用缓存
        if self.config['misc']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
            # 禁用use_cache以避免与gradient checkpointing冲突
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False
                console.print(f"[yellow]禁用 use_cache 以兼容 gradient checkpointing[/yellow]")
        
        console.print(f"[green]模型和tokenizer加载完成[/green]")
        console.print(f"模型参数量: {self.model.num_parameters():,}")
        console.print(f"词汇表大小: {len(self.tokenizer):,}")
        console.print(f"特殊token配置:")
        console.print(f"  - pad_token: {self.tokenizer.pad_token} (id: {self.tokenizer.pad_token_id})")
        console.print(f"  - bos_token: {self.tokenizer.bos_token} (id: {self.tokenizer.bos_token_id})")
        console.print(f"  - eos_token: {self.tokenizer.eos_token} (id: {self.tokenizer.eos_token_id})")
    
    def load_dataset(self, dataset_path: str):
        """加载训练数据集"""
        console.print(f"[blue]加载数据集: {dataset_path}[/blue]")
        
        dataset = load_from_disk(dataset_path)
        console.print(f"[green]数据集加载完成: {len(dataset):,} 样本[/green]")
        
        return dataset
    
    def setup_training_arguments(self, output_dir: str, run_name: str):
        """设置训练参数"""
        training_config = self.config['training']
        fsdp_config = self.config['fsdp']
        misc_config = self.config['misc']
        
        # 构建FSDP配置字典 (用于fsdp_config参数)
        fsdp_config_dict = {
            'backward_prefetch_policy': fsdp_config['fsdp_backward_prefetch'],
            'forward_prefetch': fsdp_config['fsdp_forward_prefetch'],
            'use_orig_params': fsdp_config['fsdp_use_orig_params'],
            'cpu_ram_efficient_loading': fsdp_config['fsdp_cpu_ram_efficient_loading'],
            'auto_wrap_policy': fsdp_config['fsdp_auto_wrap_policy'],
            'sharding_strategy': fsdp_config['fsdp_sharding_strategy'],
            'state_dict_type': fsdp_config['fsdp_state_dict_type'],
        }
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            num_train_epochs=training_config['num_train_epochs'],
            max_steps=training_config['max_steps'],
            warmup_ratio=training_config['warmup_ratio'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            save_total_limit=training_config['save_total_limit'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            greater_is_better=training_config['greater_is_better'],
            eval_strategy=training_config['evaluation_strategy'],
            save_strategy=training_config['save_strategy'],
            report_to=training_config['report_to'],
            run_name=run_name,
            seed=training_config['seed'],
            data_seed=training_config['data_seed'],
            dataloader_num_workers=training_config['dataloader_num_workers'],
            remove_unused_columns=training_config['remove_unused_columns'],
            label_names=training_config['label_names'],
            bf16=misc_config['bf16'],
            fp16=misc_config['fp16'],
            tf32=misc_config['tf32'],
            gradient_checkpointing=misc_config['gradient_checkpointing'],
            ddp_find_unused_parameters=misc_config['ddp_find_unused_parameters'],
            group_by_length=misc_config['group_by_length'],
            length_column_name=misc_config['length_column_name'],
            disable_tqdm=misc_config['disable_tqdm'],
            prediction_loss_only=misc_config['prediction_loss_only'],
            include_inputs_for_metrics=misc_config['include_inputs_for_metrics'],
            fsdp=fsdp_config['fsdp'],
            fsdp_config=fsdp_config_dict,
            fsdp_transformer_layer_cls_to_wrap=fsdp_config['fsdp_transformer_layer_cls_to_wrap']
        )
        
        return training_args
    
    def setup_wandb(self, run_name: str):
        """设置wandb"""
        wandb_config = self.config['wandb']
        
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config['entity'],
            name=run_name,
            tags=wandb_config['tags'],
            notes=wandb_config['notes'],
            config=self.config
        )
        
        console.print(f"[green]Wandb初始化完成: {run_name}[/green]")
    
    def train(self, dataset_path: str, output_dir: str, run_name: str = None):
        """执行训练"""
        if run_name is None:
            dataset_name = os.path.basename(dataset_path)
            run_name = f"sft-{dataset_name}"
        
        console.print(f"[bold blue]开始训练: {run_name}[/bold blue]")
        
        # 设置wandb
        self.setup_wandb(run_name)
        
        # 设置模型和tokenizer
        self.setup_model_and_tokenizer()
        
        # 加载数据集
        train_dataset = self.load_dataset(dataset_path)
        
        # 设置训练参数
        training_args = self.setup_training_arguments(output_dir, run_name)
        
        # 创建自定义数据整理器来处理text字段
        def data_collator(features):
            # 提取text字段并进行tokenization
            texts = [feature['text'] for feature in features]
            
            # 使用tokenizer进行批量tokenization
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=sft_config['max_seq_length'],
                return_tensors="pt"
            )
            
            # 对于因果语言模型，labels就是input_ids的副本
            batch['labels'] = batch['input_ids'].clone()
            
            return batch
        
        # 创建SFTTrainer
        sft_config = self.config['sft']
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            args=training_args,
            max_seq_length=sft_config['max_seq_length'],
            packing=sft_config['packing'],
            dataset_text_field=sft_config['dataset_text_field'],
            data_collator=data_collator,
            formatting_func=None  # 禁用自动格式化，使用预处理的text字段
        )
        
        console.print(f"[green]SFTTrainer创建完成[/green]")
        
        # 开始训练
        console.print(f"[bold yellow]开始训练...[/bold yellow]")
        trainer.train()
        trainer.save_model()
        # trainer.save_state()
        # 训练完成，模型已根据save_strategy自动保存
        console.print(f"[green]训练完成！模型已保存到: {output_dir}[/green]")
        
        # 结束wandb运行
        wandb.finish()
        
        console.print(f"[bold green]训练完成: {run_name}[/bold green]")
        console.print(f"模型保存路径: {output_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SFT训练脚本")
    parser.add_argument("--dataset_path", type=str, required=True, help="训练数据集路径")
    parser.add_argument("--output_dir", type=str, required=True, help="模型输出目录")
    parser.add_argument("--run_name", type=str, default=None, help="运行名称")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化训练管道
    pipeline = SFTTrainingPipeline(args.config)
    
    # 开始训练
    pipeline.train(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        run_name=args.run_name
    )

if __name__ == "__main__":
    main()