#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将 PyTorch 模型文件转换为 safetensors 格式
用于解决 PyTorch 安全漏洞问题 (CVE-2025-32434)
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    from safetensors.torch import save_file, load_file, save_model, load_model
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请安装必要的依赖:")
    print("pip install torch safetensors transformers")
    sys.exit(1)


def detect_shared_tensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, list]:
    """检测共享张量"""
    shared_tensors = {}
    tensor_to_names = {}
    
    for name, tensor in tensors.items():
        tensor_id = id(tensor.storage())
        if tensor_id not in tensor_to_names:
            tensor_to_names[tensor_id] = []
        tensor_to_names[tensor_id].append(name)
    
    for tensor_id, names in tensor_to_names.items():
        if len(names) > 1:
            shared_tensors[tensor_id] = names
    
    return shared_tensors


def resolve_shared_tensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """解决共享张量问题，保留第一个张量，其他的创建副本"""
    shared_tensors = detect_shared_tensors(tensors)
    resolved_tensors = {}
    
    if shared_tensors:
        print(f"⚠️  检测到共享张量: {list(shared_tensors.values())}")
        print("正在创建独立副本以避免共享内存问题...")
    
    processed_storages = set()
    
    for name, tensor in tensors.items():
        tensor_id = id(tensor.storage())
        
        if tensor_id in processed_storages:
            # 为共享张量创建独立副本
            resolved_tensors[name] = tensor.clone().detach()
            print(f"  - 为 {name} 创建独立副本")
        else:
            # 第一次遇到这个存储，直接使用
            resolved_tensors[name] = tensor
            processed_storages.add(tensor_id)
    
    return resolved_tensors


def convert_pytorch_to_safetensors(
    model_path: str,
    output_path: str,
    trust_remote_code: bool = False,
    max_shard_size: str = "5GB",
    use_model_api: bool = True
) -> None:
    """
    将 PyTorch 模型转换为 safetensors 格式
    
    Args:
        model_path: 输入模型路径
        output_path: 输出路径
        trust_remote_code: 是否信任远程代码
        max_shard_size: 最大分片大小
        use_model_api: 是否使用 save_model API (推荐，自动处理共享张量)
    """
    print(f"🔄 开始转换模型: {model_path}")
    print(f"📁 输出路径: {output_path}")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 方法1: 尝试使用 save_model API (推荐方法)
    if use_model_api:
        try:
            print("\n--- 方法1: 使用 save_model API (推荐) ---")
            print("正在加载完整模型...")
            
            # 检查是否为本地路径
            if os.path.isdir(model_path):
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    dtype=torch.float16,
                    device_map="cpu"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    dtype=torch.float16,
                    device_map="cpu"
                )
            
            print("正在保存为 safetensors 格式...")
            output_file = os.path.join(output_path, "model.safetensors")
            save_model(model, output_file)
            print(f"✓ 保存完成: model.safetensors")
            
            # 复制配置文件
            print("\n--- 复制配置文件 ---")
            config_files = [
                "config.json", "tokenizer.json", "tokenizer_config.json", 
                "special_tokens_map.json", "vocab.txt", "vocab.json", "merges.txt"
            ]
            
            for config_file in config_files:
                src_file = os.path.join(model_path, config_file)
                dst_file = os.path.join(output_path, config_file)
                
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"✓ 复制: {config_file}")
            
            print(f"\n🎉 转换完成! 输出目录: {output_path}")
            print("使用 save_model API 成功处理了共享张量问题。")
            return
            
        except Exception as e:
            print(f"⚠️  save_model API 失败: {e}")
            print("回退到手动处理方法...")
            use_model_api = False
    
    # 方法2: 手动处理权重文件
    print("\n--- 方法2: 手动处理权重文件 ---")
    
    # 1. 复制配置文件
    print("\n--- 步骤 1: 复制配置文件 ---")
    config_files = [
        "config.json", "tokenizer.json", "tokenizer_config.json", 
        "special_tokens_map.json", "vocab.txt", "vocab.json", "merges.txt",
        "mariana_config.json", "configuration_thoth.py", "modeling_thoth.py"
    ]
    
    for config_file in config_files:
        src_file = os.path.join(model_path, config_file)
        dst_file = os.path.join(output_path, config_file)
        
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"✓ 复制: {config_file}")
    
    # 2. 查找所有 PyTorch 权重文件
    print("\n--- 步骤 2: 查找 PyTorch 权重文件 ---")
    pytorch_files = []
    
    # 查找不同格式的权重文件
    weight_patterns = [
        "pytorch_model.bin",
        "pytorch_model-*.bin", 
        "model.safetensors",
        "model-*.safetensors"
    ]
    
    for pattern in weight_patterns:
        if "*" in pattern:
            # 处理分片文件
            import glob
            files = glob.glob(os.path.join(model_path, pattern))
            pytorch_files.extend(files)
        else:
            file_path = os.path.join(model_path, pattern)
            if os.path.exists(file_path):
                pytorch_files.append(file_path)
    
    if not pytorch_files:
        print("❌ 未找到任何权重文件")
        return
    
    print(f"找到 {len(pytorch_files)} 个权重文件:")
    for f in pytorch_files:
        print(f"  - {os.path.basename(f)}")
    
    # 3. 加载和合并所有权重
    print("\n--- 步骤 3: 加载权重文件 ---")
    all_tensors = {}
    
    for weight_file in pytorch_files:
        print(f"加载: {os.path.basename(weight_file)}")
        
        try:
            if weight_file.endswith('.safetensors'):
                # 如果已经是 safetensors 格式
                tensors = load_file(weight_file)
            else:
                # 加载 PyTorch 格式，使用安全模式
                try:
                    # 尝试使用 weights_only=True (需要 PyTorch 2.6+)
                    tensors = torch.load(weight_file, map_location='cpu', weights_only=True)
                except TypeError:
                    # 如果不支持 weights_only 参数，使用传统方法（有安全风险）
                    print("⚠️  警告: 使用传统加载方法，存在安全风险")
                    tensors = torch.load(weight_file, map_location='cpu')
            
            # 合并张量
            for key, tensor in tensors.items():
                if key in all_tensors:
                    print(f"⚠️  警告: 重复的键 {key}，将被覆盖")
                all_tensors[key] = tensor
                
        except Exception as e:
            print(f"❌ 加载文件失败 {weight_file}: {e}")
            continue
    
    if not all_tensors:
        print("❌ 没有成功加载任何张量")
        return
    
    print(f"✓ 总共加载了 {len(all_tensors)} 个张量")
    
    # 3.5. 解决共享张量问题
    print("\n--- 步骤 3.5: 解决共享张量问题 ---")
    all_tensors = resolve_shared_tensors(all_tensors)
    
    # 4. 计算总大小并决定是否分片
    print("\n--- 步骤 4: 计算模型大小 ---")
    total_size = 0
    for tensor in all_tensors.values():
        total_size += tensor.numel() * tensor.element_size()
    
    total_size_gb = total_size / (1024**3)
    print(f"模型总大小: {total_size_gb:.2f} GB")
    
    # 解析最大分片大小
    max_size_bytes = parse_size(max_shard_size)
    
    if total_size <= max_size_bytes:
        # 单文件保存
        print("📦 保存为单个 safetensors 文件")
        output_file = os.path.join(output_path, "model.safetensors")
        save_file(all_tensors, output_file)
        print(f"✓ 保存完成: model.safetensors")
        
        # 创建索引文件
        create_single_file_index(output_path, total_size)
        
    else:
        # 分片保存
        print(f"📦 模型过大，将分片保存 (每片最大 {max_shard_size})")
        shard_files = save_sharded_safetensors(all_tensors, output_path, max_size_bytes)
        
        # 创建分片索引文件
        create_sharded_index(output_path, shard_files, all_tensors)
    
    print(f"\n🎉 转换完成! 输出目录: {output_path}")
    print("现在可以安全地使用 safetensors 格式加载模型了。")


def parse_size(size_str: str) -> int:
    """解析大小字符串，如 '5GB', '1TB' 等"""
    size_str = size_str.upper().strip()
    
    if size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024**3)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024**2)
    elif size_str.endswith('TB'):
        return int(float(size_str[:-2]) * 1024**4)
    else:
        return int(size_str)


def save_sharded_safetensors(tensors: Dict[str, torch.Tensor], output_path: str, max_size: int) -> list:
    """将张量分片保存为多个 safetensors 文件"""
    shard_files = []
    current_shard = {}
    current_size = 0
    shard_index = 1
    
    for name, tensor in tensors.items():
        tensor_size = tensor.numel() * tensor.element_size()
        
        # 如果当前分片加上这个张量会超过限制，保存当前分片
        if current_size + tensor_size > max_size and current_shard:
            shard_filename = f"model-{shard_index:05d}-of-{len(tensors):05d}.safetensors"
            shard_path = os.path.join(output_path, shard_filename)
            
            save_file(current_shard, shard_path)
            shard_files.append(shard_filename)
            print(f"✓ 保存分片: {shard_filename} ({current_size / 1024**2:.1f} MB)")
            
            current_shard = {}
            current_size = 0
            shard_index += 1
        
        current_shard[name] = tensor
        current_size += tensor_size
    
    # 保存最后一个分片
    if current_shard:
        shard_filename = f"model-{shard_index:05d}-of-{len(tensors):05d}.safetensors"
        shard_path = os.path.join(output_path, shard_filename)
        
        save_file(current_shard, shard_path)
        shard_files.append(shard_filename)
        print(f"✓ 保存分片: {shard_filename} ({current_size / 1024**2:.1f} MB)")
    
    return shard_files


def create_single_file_index(output_path: str, total_size: int) -> None:
    """为单文件模型创建索引"""
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {"model.safetensors": "model.safetensors"}
    }
    
    index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def create_sharded_index(output_path: str, shard_files: list, tensors: Dict[str, torch.Tensor]) -> None:
    """为分片模型创建索引文件"""
    weight_map = {}
    total_size = 0
    
    # 重新计算每个张量在哪个分片中
    current_shard_idx = 0
    current_size = 0
    max_size = parse_size("5GB")  # 默认分片大小
    
    for name, tensor in tensors.items():
        tensor_size = tensor.numel() * tensor.element_size()
        total_size += tensor_size
        
        if current_size + tensor_size > max_size and current_shard_idx < len(shard_files) - 1:
            current_shard_idx += 1
            current_size = 0
        
        weight_map[name] = shard_files[current_shard_idx]
        current_size += tensor_size
    
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }
    
    index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="将 PyTorch 模型转换为 safetensors 格式")
    parser.add_argument("--input_path", type=str, default="/mnt/hdfs/selection/tt_stage2_model/general",
                       help="输入模型路径")
    parser.add_argument("--output_path", type=str, default="/mnt/hdfs/selection/tt_stage2_model/general_safe",
                       help="输出路径")
    parser.add_argument("--trust_remote_code", default=True,
                       help="是否信任远程代码")
    parser.add_argument("--max_shard_size", type=str, default="5GB",
                       help="最大分片大小 (例如: 5GB, 1TB)")
    parser.add_argument("--force", default=True,
                       help="强制覆盖输出目录")
    parser.add_argument("--use_model_api", action="store_true", default=True,
                       help="使用 save_model API (推荐，自动处理共享张量)")
    parser.add_argument("--manual_mode", action="store_true",
                       help="强制使用手动处理模式")
    
    args = parser.parse_args()
    
    # 检查输入路径 (允许 Hugging Face 模型名称)
    if not os.path.exists(args.input_path) and "/" not in args.input_path:
        # 可能是 Hugging Face 模型名称，尝试验证
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(args.input_path, trust_remote_code=args.trust_remote_code)
            print(f"✓ 检测到 Hugging Face 模型: {args.input_path}")
        except Exception as e:
            print(f"❌ 输入路径不存在且不是有效的 Hugging Face 模型: {args.input_path}")
            print(f"错误: {e}")
            sys.exit(1)
    elif not os.path.exists(args.input_path):
        print(f"❌ 输入路径不存在: {args.input_path}")
        sys.exit(1)
    
    # 检查输出路径
    if os.path.exists(args.output_path) and not args.force:
        print(f"❌ 输出路径已存在: {args.output_path}")
        print("使用 --force 参数强制覆盖")
        sys.exit(1)
    
    try:
        # 如果指定了手动模式，则不使用 model API
        use_model_api = not args.manual_mode
        
        convert_pytorch_to_safetensors(
            model_path=args.input_path,
            output_path=args.output_path,
            trust_remote_code=args.trust_remote_code,
            max_shard_size=args.max_shard_size,
            use_model_api=use_model_api
        )
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()