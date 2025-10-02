#!/usr/bin/env python3
"""
调试张量转换问题的简单脚本
"""

import torch
from transformers import AutoTokenizer

def test_tokenization(model_name="microsoft/DialoGPT-medium"):
    """测试分词过程"""
    print(f"测试模型: {model_name}")
    
    # 你的数据样本
    sample_text = '<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n💕<|im_end|>\n'
    
    print(f"原始文本: {repr(sample_text)}")
    print(f"文本类型: {type(sample_text)}")
    print(f"文本长度: {len(sample_text)}")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"设置 pad_token: {tokenizer.pad_token}")
        
        # 移除token_type_ids支持
        if hasattr(tokenizer, 'model_input_names'):
            if 'token_type_ids' in tokenizer.model_input_names:
                tokenizer.model_input_names = [name for name in tokenizer.model_input_names if name != 'token_type_ids']
                print("移除了 token_type_ids 支持")
        
        print(f"Tokenizer model_input_names: {tokenizer.model_input_names}")
        
        # 测试分词
        print("\n=== 测试分词 ===")
        
        # 方法1: 基本分词
        tokens = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        print(f"分词结果键: {list(tokens.keys())}")
        for key, value in tokens.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # 方法2: 批量分词
        batch_texts = [sample_text, sample_text]  # 模拟批量数据
        batch_tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        print(f"\n批量分词结果键: {list(batch_tokens.keys())}")
        for key, value in batch_tokens.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        print("\n✅ 分词测试成功!")
        return True
        
    except Exception as e:
        print(f"\n❌ 分词测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_collator():
    """测试数据整理器"""
    print("\n=== 测试数据整理器 ===")
    
    try:
        from transformers import DataCollatorForLanguageModeling, AutoTokenizer
        
        # 使用一个通用的tokenizer进行测试
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 移除token_type_ids
        if hasattr(tokenizer, 'model_input_names'):
            if 'token_type_ids' in tokenizer.model_input_names:
                tokenizer.model_input_names = [name for name in tokenizer.model_input_names if name != 'token_type_ids']
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        # 模拟数据
        sample_text = '<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n💕<|im_end|>\n'
        
        # 分词
        tokenized = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # 转换为列表格式（模拟数据集格式）
        batch = []
        for i in range(len(tokenized['input_ids'])):
            item = {}
            for key, value in tokenized.items():
                item[key] = value[i]
            batch.append(item)
        
        # 测试数据整理器
        collated = data_collator(batch)
        print(f"整理后的数据键: {list(collated.keys())}")
        for key, value in collated.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        print("✅ 数据整理器测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 数据整理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始调试张量转换问题...")
    
    # 测试不同的tokenizer
    models_to_test = [
        "gpt2",  # 基础测试
        "microsoft/DialoGPT-medium",  # 对话模型
    ]
    
    for model in models_to_test:
        print(f"\n{'='*50}")
        try:
            success = test_tokenization(model)
            if not success:
                print(f"模型 {model} 测试失败")
        except Exception as e:
            print(f"无法测试模型 {model}: {e}")
    
    # 测试数据整理器
    test_data_collator()
    
    print(f"\n{'='*50}")
    print("调试完成!")