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

# 修补 DynamicCache 兼容性问题
def patch_dynamic_cache():
    """为 DynamicCache 添加缺失的方法以修复兼容性问题"""
    try:
        from transformers.cache_utils import DynamicCache
        
        def get_usable_length(self, seq_length=None):
            """获取可用的缓存长度"""
            if not hasattr(self, 'layers') or len(self.layers) == 0:
                return 0
            # 使用现有的 get_seq_length 方法
            if hasattr(self, 'get_seq_length'):
                return self.get_seq_length()
            return 0
        
        def get_seq_length(self, layer_idx=None):
            """获取序列长度（兼容性方法）"""
            if layer_idx is not None and layer_idx < len(self.key_cache):
                if self.key_cache[layer_idx] is not None:
                    return self.key_cache[layer_idx].shape[-2]
            return self.get_usable_length()
        
        def get_max_length(self):
            """获取最大长度（兼容性方法）"""
            return None  # DynamicCache 没有最大长度限制
        
        # 添加缺失的方法
        methods_to_add = [
            ('get_usable_length', get_usable_length),
            ('get_seq_length', get_seq_length),
            ('get_max_length', get_max_length),
        ]
        
        patched_methods = []
        for method_name, method_func in methods_to_add:
            if not hasattr(DynamicCache, method_name):
                setattr(DynamicCache, method_name, method_func)
                patched_methods.append(method_name)
        
        if patched_methods:
            console.print(f"[yellow]已修补 DynamicCache 方法: {', '.join(patched_methods)}[/yellow]")
        else:
            console.print("[blue]DynamicCache 已包含所有必要方法[/blue]")
            
    except ImportError:
        console.print("[yellow]无法导入 DynamicCache，跳过修补[/yellow]")

# 应用修补
patch_dynamic_cache()

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
        
        # 确保tokenizer不返回token_type_ids（ThothForCausalLM不支持）
        if hasattr(self.tokenizer, 'model_input_names'):
            if 'token_type_ids' in self.tokenizer.model_input_names:
                self.tokenizer.model_input_names = [name for name in self.tokenizer.model_input_names if name != 'token_type_ids']
                console.print(f"[yellow]从tokenizer中移除token_type_ids支持[/yellow]")
        
        # 设置tokenizer的默认行为
        self.tokenizer.padding_side = "right"
        console.print(f"[blue]Tokenizer配置: padding_side={self.tokenizer.padding_side}[/blue]")
        
        # 加载模型
        model_kwargs = {
            'trust_remote_code': self.config['model']['trust_remote_code'],
            'dtype': torch.bfloat16 if self.config['misc']['bf16'] else torch.float16,
            'device_map': None,  # FSDP会处理设备分配
        }
        
        # 添加 attention implementation 配置
        if 'attn_implementation' in self.config['model']:
            model_kwargs['attn_implementation'] = self.config['model']['attn_implementation']
            console.print(f"[blue]使用 attention implementation: {self.config['model']['attn_implementation']}[/blue]")
        
        if self.config['model']['use_flash_attention_2']:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['model_name'],
            **model_kwargs
        )
        
        # 设置 use_cache 配置
        if 'use_cache' in self.config['model']:
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = self.config['model']['use_cache']
                console.print(f"[blue]设置 use_cache: {self.config['model']['use_cache']}[/blue]")
        
        # 默认禁用 use_cache 以避免兼容性问题
        if hasattr(self.model.config, 'use_cache') and self.model.config.use_cache:
            self.model.config.use_cache = False
            console.print(f"[yellow]为了兼容性，强制禁用 use_cache[/yellow]")
        
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
        
        # 检查并修复数据格式
        dataset = self.preprocess_dataset(dataset)
        
        return dataset
    
    def preprocess_dataset(self, dataset):
        """预处理数据集，确保text字段格式正确"""
        console.print(f"[blue]预处理数据集...[/blue]")
        
        # 首先检查原始数据结构
        sample = dataset[0]
        console.print(f"[yellow]原始数据结构检查:[/yellow]")
        console.print(f"  - 样本字段: {list(sample.keys())}")
        console.print(f"  - text字段类型: {type(sample.get('text', 'NOT_FOUND'))}")
        if 'text' in sample:
            text_value = sample['text']
            if isinstance(text_value, list):
                console.print(f"  - text是列表，长度: {len(text_value)}")
                if len(text_value) > 0:
                    console.print(f"  - 第一个元素类型: {type(text_value[0])}")
                    console.print(f"  - 第一个元素内容: {str(text_value[0])[:100]}...")
            else:
                console.print(f"  - text内容预览: {str(text_value)[:100]}...")
        
        def process_text_field(example):
            text = example.get('text', '')
            original_type = type(text)
            
            # 如果text是列表，尝试连接成字符串
            if isinstance(text, list):
                if len(text) > 0 and isinstance(text[0], str):
                    # 如果是字符串列表，连接成单个字符串
                    text = ' '.join(text)
                elif len(text) > 0 and isinstance(text[0], dict):
                    # 如果是字典列表，尝试提取文本内容
                    text_parts = []
                    for item in text:
                        if isinstance(item, dict):
                            # 尝试常见的文本字段名
                            for key in ['text', 'content', 'message', 'input', 'output']:
                                if key in item and isinstance(item[key], str):
                                    text_parts.append(item[key])
                                    break
                        elif isinstance(item, str):
                            text_parts.append(item)
                    text = ' '.join(text_parts)
                else:
                    # 其他情况，转换为字符串
                    text = str(text)
            elif not isinstance(text, str):
                # 如果不是字符串，转换为字符串
                text = str(text)
            
            # 确保text不为空
            if not text or text.strip() == '':
                text = "Empty text"
            
            # 清理可能的特殊字符和多余空格
            text = ' '.join(text.split())
            
            example['text'] = text
            return example
        
        # 应用预处理
        dataset = dataset.map(
            process_text_field, 
            desc="预处理文本字段",
            num_proc=8  # 使用多线程处理，避免CPU内存不足
        )
        
        # 验证处理结果
        sample = dataset[0]
        console.print(f"[green]预处理完成，样本文本类型: {type(sample['text'])}, 长度: {len(sample['text'])}[/green]")
        console.print(f"[yellow]样本预览: {sample['text'][:100]}...[/yellow]")
        
        return dataset
    
    def validate_dataset(self, dataset):
        """验证数据集格式"""
        console.print(f"[blue]验证数据集格式...[/blue]")
        
        # 检查数据集是否为空
        if len(dataset) == 0:
            raise ValueError("数据集为空")
        
        # 检查前几个样本的格式
        text_field = self.config['sft']['dataset_text_field']
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            
            # 检查text字段是否存在
            if text_field not in sample:
                raise ValueError(f"样本 {i} 缺少 '{text_field}' 字段")
            
            # 检查text字段是否为字符串
            text = sample[text_field]
            if not isinstance(text, str):
                raise ValueError(f"样本 {i} 的 '{text_field}' 字段不是字符串类型: {type(text)}")
            
            # 检查text是否为空
            if not text.strip():
                console.print(f"[yellow]警告: 样本 {i} 的文本为空[/yellow]")
        
        console.print(f"[green]数据集格式验证通过[/green]")
    
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
        
        # 添加activation_checkpointing和transformer_layer_cls_to_wrap到fsdp_config
        if 'activation_checkpointing' in fsdp_config:
            fsdp_config_dict['activation_checkpointing'] = fsdp_config['activation_checkpointing']
            console.print(f"[green]启用FSDP activation_checkpointing: {fsdp_config['activation_checkpointing']}[/green]")
        if 'transformer_layer_cls_to_wrap' in fsdp_config:
            fsdp_config_dict['transformer_layer_cls_to_wrap'] = fsdp_config['transformer_layer_cls_to_wrap']
            console.print(f"[green]FSDP transformer_layer_cls_to_wrap: {fsdp_config['transformer_layer_cls_to_wrap']}[/green]")
        
        # 显示gradient_checkpointing状态
        console.print(f"[yellow]TrainingArguments gradient_checkpointing: {misc_config['gradient_checkpointing']}[/yellow]")
        
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
            # 移除deprecated的fsdp_transformer_layer_cls_to_wrap，现在使用fsdp_config中的transformer_layer_cls_to_wrap
        )
        
        return training_args
    
    def setup_wandb(self, run_name: str):
        """设置wandb"""
        wandb_config = self.config['wandb']
        
        # 设置离线模式（如果配置了的话）
        if wandb_config.get('offline', False):
            os.environ['WANDB_MODE'] = 'offline'
            console.print("[yellow]Wandb 离线模式已启用，不会上传到云端[/yellow]")
        
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config['entity'],
            name=run_name,
            tags=wandb_config['tags'],
            notes=wandb_config['notes'],
            config=self.config
        )
        
        if wandb_config.get('offline', False):
            console.print(f"[green]Wandb初始化完成（离线模式）: {run_name}[/green]")
        else:
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
        
        # 验证数据集格式
        self.validate_dataset(train_dataset)
        
        # 创建SFTTrainer
        sft_config = self.config['sft']
        console.print(f"[blue]创建SFTTrainer...[/blue]")
        
        try:
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                args=training_args,
                max_seq_length=sft_config['max_seq_length'],
                packing=sft_config['packing'],
                dataset_text_field=sft_config['dataset_text_field'],
                dataset_num_proc=training_args.dataloader_num_workers,  # 使用多进程加速数据处理
                formatting_func=None  # 禁用自动格式化，使用预处理的text字段
            )
        except Exception as e:
            console.print(f"[red]SFTTrainer创建失败: {e}[/red]")
            console.print(f"[yellow]尝试调试数据集格式...[/yellow]")
            
            # 输出更多调试信息
            sample = train_dataset[0]
            console.print(f"样本字段: {list(sample.keys())}")
            text_field = sft_config['dataset_text_field']
            if text_field in sample:
                text_value = sample[text_field]
                console.print(f"文本字段 '{text_field}' 类型: {type(text_value)}")
                console.print(f"文本内容: {str(text_value)[:200]}...")
            
            raise e
        
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