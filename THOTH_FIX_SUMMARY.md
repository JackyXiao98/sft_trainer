# Thoth 模型 DynamicCache 兼容性修复

## 问题描述

在运行 `training_config_thoth.yaml` 配置时遇到以下错误：

```
AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'
```

错误发生在 Thoth 模型的前向传播过程中：
```python
past_key_values_length = past_key_values.get_usable_length(seq_length)
```

## 根本原因

Thoth 模型的代码使用了 `DynamicCache.get_usable_length()` 方法，但该方法在当前版本的 transformers 库中不存在。这是一个版本兼容性问题。

## 解决方案

### 1. 配置文件修复

在 `configs/training_config_thoth.yaml` 中添加了以下配置：

```yaml
model:
  use_cache: false  # 禁用缓存以避免 DynamicCache 兼容性问题
  attn_implementation: "flash_attention_2"  # 明确指定注意力实现
```

### 2. 训练代码修复

在 `src/train.py` 中添加了 DynamicCache 修补代码：

```python
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
        
        def get_max_length(self):
            """获取最大长度（兼容性方法）"""
            return None  # DynamicCache 没有最大长度限制
        
        # 添加缺失的方法
        if not hasattr(DynamicCache, 'get_usable_length'):
            DynamicCache.get_usable_length = get_usable_length
        if not hasattr(DynamicCache, 'get_max_length'):
            DynamicCache.get_max_length = get_max_length
            
    except ImportError:
        pass

# 在模块加载时自动应用修补
patch_dynamic_cache()
```

### 3. 模型配置强化

在模型加载时强制设置 `use_cache=False`：

```python
# 设置 use_cache 配置
if 'use_cache' in self.config['model']:
    if hasattr(self.model.config, 'use_cache'):
        self.model.config.use_cache = self.config['model']['use_cache']

# 默认禁用 use_cache 以避免兼容性问题
if hasattr(self.model.config, 'use_cache') and self.model.config.use_cache:
    self.model.config.use_cache = False
```

## 修复验证

修复后的代码已通过测试验证：

1. ✅ DynamicCache 成功添加了 `get_usable_length` 方法
2. ✅ DynamicCache 成功添加了 `get_max_length` 方法  
3. ✅ 方法调用正常工作
4. ✅ 训练模块导入成功

## 使用方法

现在可以正常运行 Thoth 模型训练：

```bash
python src/train.py configs/training_config_thoth.yaml \
    --dataset_path <数据集路径> \
    --output_dir <输出目录>
```

## 注意事项

1. 这个修复是在运行时动态应用的，不会影响 transformers 库的原始代码
2. 修复代码会在每次导入 `src/train.py` 时自动执行
3. 如果 transformers 库更新并原生支持这些方法，修复代码会自动跳过
4. 建议在生产环境中固定 transformers 库版本以确保兼容性

## 相关文件

- `configs/training_config_thoth.yaml` - 更新了模型配置
- `src/train.py` - 添加了 DynamicCache 修补代码