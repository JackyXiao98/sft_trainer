#!/usr/bin/env python3
"""
测试 SFT 特定的数据处理问题
"""

import torch
from transformers import AutoTokenizer
from datasets import Dataset

def create_mock_dataset():
    """创建模拟数据集"""
    # 基于你提供的真实数据样本
    sample_data = {
        'text': '<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n💕<|im_end|>\n',
        'messages': [{'content': '', 'role': 'user'}, {'content': '💕', 'role': 'assistant'}],
        'original_comment_id': '7479649524542620434',
        'interval': '[0.0, 0.1)',
        'dataset': 'tiktok',
        'category': 'comments',
        'language': 'en',
        'language_score': 0.7255634665489197,
        'token_num': 2
    }
    
    # 创建多个样本
    data = []
    for i in range(5):
        sample = sample_data.copy()
        sample['original_comment_id'] = f'test_id_{i}'
        data.append(sample)
    
    return Dataset.from_list(data)

def test_tokenizer_with_dataset():
    """测试 tokenizer 与数据集的交互"""
    print("=== 测试 Tokenizer 与数据集交互 ===")
    
    try:
        # 创建模拟数据集
        dataset = create_mock_dataset()
        print(f"数据集大小: {len(dataset)}")
        print(f"数据集字段: {list(dataset[0].keys())}")
        print(f"文本样本: {dataset[0]['text'][:100]}...")
        
        # 使用 GPT2 tokenizer 进行测试
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 移除 token_type_ids
        if hasattr(tokenizer, 'model_input_names'):
            if 'token_type_ids' in tokenizer.model_input_names:
                tokenizer.model_input_names = [name for name in tokenizer.model_input_names if name != 'token_type_ids']
        
        print(f"Tokenizer model_input_names: {tokenizer.model_input_names}")
        
        # 测试单个样本分词
        sample_text = dataset[0]['text']
        print(f"\n原始文本类型: {type(sample_text)}")
        print(f"原始文本内容: {repr(sample_text[:50])}...")
        
        # 分词测试
        tokens = tokenizer(
            sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        print(f"分词结果: {list(tokens.keys())}")
        for key, value in tokens.items():
            print(f"  {key}: {value.shape}, {value.dtype}")
        
        # 测试批量分词
        batch_texts = [item['text'] for item in dataset]
        print(f"\n批量文本数量: {len(batch_texts)}")
        print(f"批量文本类型: {[type(t) for t in batch_texts[:3]]}")
        
        batch_tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        print(f"批量分词结果: {list(batch_tokens.keys())}")
        for key, value in batch_tokens.items():
            print(f"  {key}: {value.shape}, {value.dtype}")
        
        # 测试数据集映射
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding=False,  # 不在这里padding
                truncation=True,
                max_length=512
            )
        
        print(f"\n=== 测试数据集映射 ===")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        print(f"映射后数据集字段: {list(tokenized_dataset[0].keys())}")
        sample_tokenized = tokenized_dataset[0]
        for key, value in sample_tokenized.items():
            print(f"  {key}: {type(value)}, 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}")
        
        print("✅ Tokenizer 与数据集交互测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_collator_with_dataset():
    """测试数据整理器与数据集"""
    print(f"\n=== 测试数据整理器与数据集 ===")
    
    try:
        from transformers import DataCollatorForLanguageModeling
        
        # 创建数据集和tokenizer
        dataset = create_mock_dataset()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 移除 token_type_ids
        if hasattr(tokenizer, 'model_input_names'):
            if 'token_type_ids' in tokenizer.model_input_names:
                tokenizer.model_input_names = [name for name in tokenizer.model_input_names if name != 'token_type_ids']
        
        # 分词数据集
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding=False,
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 创建数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # 测试批量整理
        batch_size = 2
        batch = [tokenized_dataset[i] for i in range(batch_size)]
        
        print(f"批量数据样本数: {len(batch)}")
        print(f"第一个样本字段: {list(batch[0].keys())}")
        
        collated = data_collator(batch)
        print(f"整理后字段: {list(collated.keys())}")
        for key, value in collated.items():
            print(f"  {key}: {value.shape}, {value.dtype}")
        
        print("✅ 数据整理器测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 数据整理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始 SFT 特定测试...")
    
    success1 = test_tokenizer_with_dataset()
    success2 = test_data_collator_with_dataset()
    
    if success1 and success2:
        print(f"\n✅ 所有测试通过!")
    else:
        print(f"\n❌ 部分测试失败!")