#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoConfig

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="检查模型的 FSDP Transformer Layer 类名")
    parser.add_argument("--model_path", type=str, 
                       default="/mnt/hdfs/selection/tt_stage2_model/domain",
                       help="模型路径")
    parser.add_argument("--trust_remote_code", action="store_true", 
                       help="是否信任远程代码（用于自定义模型）")
    
    args = parser.parse_args()
    model_path = args.model_path

    print(f"--- 步骤 1: 检查模型路径 ---")
    print(f"模型: {model_path}")

    # 检查是否为本地路径
    is_local_path = os.path.isdir(model_path)
    if is_local_path:
        print("✓ 检测到本地模型路径")
    else:
        print("✓ 检测到 Hugging Face 模型名称，将从远程下载")
    
    print("模型路径检查通过！")

    # --- 2. 尝试加载配置文件 ---
    try:
        print("\n--- 步骤 2: 加载模型配置 ---")
        config = AutoConfig.from_pretrained(
            model_path, 
            trust_remote_code=args.trust_remote_code
        )
        print(f"模型类型: {config.model_type}")
        print(f"架构: {config.architectures if hasattr(config, 'architectures') else '未知'}")
        print("配置加载成功！")
    except Exception as e:
        print(f"❌ 加载配置时出错: {e}")
        print("请检查模型目录中是否存在 'config.json' 文件。")
        sys.exit(1)

    # --- 3. 加载模型 ---
    try:
        print(f"\n--- 步骤 3: 从 '{model_path}' 加载模型 ---")
        print("注意: 仅加载模型结构，不加载权重到GPU以节省内存...")
        
        # 使用 AutoModelForCausalLM 自动识别模型类型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
            device_map="cpu"  # 强制使用CPU以避免GPU内存问题
        )
        print("模型结构加载成功！")
        print(f"模型类: {model.__class__.__name__}")
        
    except Exception as e:
        print(f"❌ 加载模型时出错: {e}")
        print("请检查模型文件是否完整，以及权重文件是否存在。")
        print("如果是自定义模型，请尝试添加 --trust_remote_code 参数。")
        sys.exit(1)


    # --- 4. 查找 Transformer Block 类名 ---
    print("\n--- 步骤 4: 查找 FSDP Transformer Block 类名 ---")

    def find_transformer_block_class(model):
        """
        递归查找模型中的 Transformer Block 类名
        """
        # 常见的 Transformer Block 类名模式
        common_patterns = [
            'TransformerBlock', 'Block', 'Layer', 'DecoderLayer', 
            'EncoderLayer', 'ThothBlock', 'ThothLayer', 'QWenBlock',
            'LlamaDecoderLayer', 'GPTBlock', 'AttentionBlock'
        ]
        
        def search_modules(module, path="", depth=0):
             results = []
             for name, child in module.named_children():
                 current_path = f"{path}.{name}" if path else name
                 class_name = child.__class__.__name__
                 
                 # 检查是否匹配常见模式
                 for pattern in common_patterns:
                     if pattern.lower() in class_name.lower():
                         # 只添加主要的 transformer block，避免子模块
                         if depth <= 2 and any(main_pattern in class_name for main_pattern in 
                                             ['Block', 'Layer', 'DecoderLayer', 'EncoderLayer']):
                             results.append((current_path, class_name, child))
                         break
                 
                 # 递归搜索子模块，但限制深度
                 if depth < 3:
                     results.extend(search_modules(child, current_path, depth + 1))
             
             return results
        
        return search_modules(model)

    # 查找所有可能的 Transformer Block
    transformer_blocks = find_transformer_block_class(model)

    if transformer_blocks:
         # 获取唯一的类名
         unique_classes = {}
         for path, class_name, module in transformer_blocks:
             if class_name not in unique_classes:
                 unique_classes[class_name] = (path, module)
         
         print("🎉 找到以下可能的 Transformer Block 类:")
         for i, (class_name, (example_path, module)) in enumerate(unique_classes.items(), 1):
             print(f"  {i}. 类名: {class_name}")
             print(f"     示例路径: {example_path}")
             print(f"     模块类型: {type(module).__name__}")
             print()
         
         # 找到最可能的 transformer block（通常包含 'Block' 或 'Layer'）
         main_transformer_classes = [name for name in unique_classes.keys() 
                                   if any(pattern in name for pattern in ['Block', 'DecoderLayer', 'EncoderLayer'])]
         
         if main_transformer_classes:
             recommended_class = main_transformer_classes[0]
             recommended_path = unique_classes[recommended_class][0]
         else:
             # 如果没找到主要的，就用第一个
             recommended_class = list(unique_classes.keys())[0]
             recommended_path = unique_classes[recommended_class][0]
         
         print(f"🔧 推荐用于 FSDP 的 Transformer Block 类名: {recommended_class}")
         print(f"🔧 推荐的模块路径: {recommended_path}")
         
         # 输出 FSDP 配置建议
         print(f"\n📋 FSDP 配置建议:")
         print(f"   auto_wrap_policy: transformer_auto_wrap_policy")
         print(f"   transformer_layer_cls_to_wrap: {recommended_class}")
        
    else:
        print("❌ 未找到明显的 Transformer Block 类。")
        print("让我们查看模型的整体结构:")
        print("\n模型结构概览:")
        for name, module in model.named_children():
            print(f"  - {name}: {module.__class__.__name__}")
        
        print("\n请手动检查模型结构以确定正确的 layer 类名。")

    print("\n--- 完成! ---")
    print("现在你可以在 FSDP 配置中使用找到的类名。")

if __name__ == "__main__":
    main()