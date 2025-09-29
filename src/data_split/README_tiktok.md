# TikTok评论数据集构建器

## 概述

`tiktok_comment_data_builder.py` 是一个专门用于处理TikTok评论数据的构建器，继承自基础的 `DataBuilder` 类。它能够处理 `/mnt/hdfs/selection/tiktok_cmt` 目录下的10个区间数据集，将原始的 `doc` 字段转换为SFT（Supervised Fine-Tuning）训练格式。

## 功能特性

### 1. 数据源处理
- 处理10个区间数据集：`[0.0, 0.1)` 到 `[0.9, 1.0)`
- 每个区间包含预处理过的 `train` 和 `val` 目录
- 自动加载目录中的所有parquet文件并合并

### 2. 数据格式转换
- 将原始的 `doc` 字段（纯文本）转换为SFT训练格式
- 生成用户-助手对话格式的训练数据
- 保留原始数据的元信息（comment_id, interval, language等）

### 3. 数据集变体生成
- 生成1个全量数据集（所有10个区间的组合）
- 为每个区间生成4种变体扰动：
  - `1/3` 变体：目标区间的1/3 + 其他所有区间
  - `1/2` 变体：目标区间的1/2 + 其他所有区间  
  - `2x` 变体：目标区间的2倍 + 其他所有区间
  - `3x` 变体：目标区间的3倍 + 其他所有区间
- 总计生成41个训练数据集（1 + 10*4）

## 数据格式示例

### 原始数据格式
```json
{
  "comment_id": "7473130329534235400",
  "parent_id": "7473002417836150034",
  "doc": "这个视频太有趣了！😂",
  "interval": "[0.0, 0.1)",
  "language": "zh",
  "token_num": 19
}
```

### 转换后的SFT格式
```json
{
  "text": "user: 请分析这条TikTok评论的内容和情感：\n这个视频太有趣了！😂\nassistant: 这是一条TikTok评论。评论内容为：这个视频太有趣了！😂\n\n这条评论表达了用户的观点和情感。",
  "messages": [
    {
      "role": "user",
      "content": "请分析这条TikTok评论的内容和情感：\n这个视频太有趣了！😂"
    },
    {
      "role": "assistant", 
      "content": "这是一条TikTok评论。评论内容为：这个视频太有趣了！😂\n\n这条评论表达了用户的观点和情感。"
    }
  ],
  "original_comment_id": "7473130329534235400",
  "interval": "[0.0, 0.1)",
  "language": "zh",
  "token_num": 19
}
```

## 使用方法

### 1. 直接运行
```bash
cd /Users/bytedance/Desktop/Github/sft_trainer
python src/data_split/tiktok_comment_data_builder.py
```

### 2. 作为模块导入
```python
from src.data_split.tiktok_comment_data_builder import TikTokCommentDataBuilder

# 创建构建器实例
builder = TikTokCommentDataBuilder()

# 构建所有数据集
builder.build_all_datasets()
```

### 3. 自定义配置
```python
# 使用自定义配置文件
builder = TikTokCommentDataBuilder(config_path="custom_config.yaml")
```

## 输出结构

构建完成后，数据集将保存在以下目录：

```
data/
├── tiktok_training/
│   ├── tiktok_full_dataset/
│   ├── tiktok_00_01_1_3/
│   ├── tiktok_00_01_1_2/
│   ├── tiktok_00_01_2x/
│   ├── tiktok_00_01_3x/
│   ├── tiktok_01_02_1_3/
│   └── ... (共41个训练数据集)
└── tiktok_validation/
    ├── [0.0, 0.1)_val/
    ├── [0.1, 0.2)_val/
    └── ... (共10个验证数据集)
```

## 配置要求

### 依赖包
- `datasets`
- `transformers` 
- `pyarrow`
- `pandas`
- `rich`
- `jinja2`

### 配置文件
需要 `configs/training_config.yaml` 文件，包含以下配置：
```yaml
model:
  model_name: "Qwen/Qwen3-8B"
  trust_remote_code: true

training:
  seed: 42

data:
  token_limits:
    train_base: 660000
    validation: 1000000
```

## 核心方法说明

### `convert_doc_to_sft_format(doc, comment_id)`
将原始的doc字段转换为SFT格式的messages列表。

### `load_parquet_files_from_directory(directory_path)`
从指定目录加载所有parquet文件并合并为单个Dataset。

### `process_tiktok_dataset(dataset, dataset_name)`
处理TikTok数据集，执行格式转换和数据清理。

### `create_training_variants(base_datasets)`
创建训练变体，包括全量数据集和各种扰动变体。

## 注意事项

1. **数据路径**：确保 `/mnt/hdfs/selection/tiktok_cmt` 路径存在且可访问
2. **内存使用**：处理大型数据集时注意内存使用情况
3. **存储空间**：确保有足够的磁盘空间保存生成的数据集
4. **权限**：确保对数据目录有读取权限，对输出目录有写入权限

## 测试

运行测试脚本验证功能：
```bash
python test_tiktok_builder.py
```

## 扩展性

该构建器设计为可扩展的，可以通过以下方式进行定制：

1. **修改SFT格式**：重写 `convert_doc_to_sft_format` 方法
2. **添加数据过滤**：在 `process_tiktok_dataset` 中添加过滤逻辑
3. **自定义变体**：修改 `_create_interval_variants` 方法
4. **支持其他数据源**：重写 `load_source_datasets` 方法