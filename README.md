# SFT Scaling Law 研究项目

本项目旨在通过在不同数据量配比下进行监督微调（SFT），来探索模型的 Scaling Law。

## 1. 环境配置

本项目建议在虚拟环境中运行，以隔离依赖。

### 1.1 创建并激活虚拟环境

```bash
# 创建一个名为 venv 的虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

### 1.2 安装依赖

本项目依赖 PyTorch, Transformers, TRL 等库。为了确保 Flash Attention 2 和 FSDP 的兼容性，请确保你的环境中已正确安装 CUDA Toolkit (推荐 >= 11.8)。

```bash
# 从 requirements.txt 安装所有依赖
pip install -r requirements.txt
```

### 1.3 Weights & Biases (W&B) 配置

在运行项目之前，请先登录你的 W&B 账户：

```bash
wandb login
```

你需要在 `configs/training_config.yaml` 文件中设置你的 W&B 项目名称。

## 2. 使用说明

整个实验流程由一个脚本自动化执行，确保你已按上述步骤完成环境配置。

### 2.1 修改配置

打开 `configs/training_config.yaml` 文件，根据你的需求和硬件环境调整训练参数，例如：
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `learning_rate`
- `num_train_epochs`
- `wandb_project` (设置为你的 wandb 项目名称)

**重要：** 确保 `fsdp` 相关的配置是开启的，以便使用 FSDP。

### 2.2 运行实验

执行以下命令来启动完整的数据生成、模型训练和评估流程：

```bash
bash scripts/run_experiments.sh
```

**脚本 `run_experiments.sh` 将会执行以下步骤：**

1.  **数据准备：** 调用 `src/data_builder.py` 生成所有 13 个训练数据集和 3 个验证数据集，并保存在 `./data/` 目录下。
2.  **循环训练与评估：**
    * 遍历 13 个训练数据集。
    * 对于每个数据集，调用 `src/train.py` 进行模型训练，并将训练好的模型 checkpoint 保存在一个唯一的目录中（例如 `outputs/model_on_dataset_X`）。
    * 训练完成后，遍历 3 个验证数据集。
    * 对于每个验证集，调用 `src/evaluate.py`，使用刚刚训练好的模型计算 loss。
3.  **结果汇总：** 所有评估结果（loss 值）将被汇总并保存在项目根目录下的 `results.csv` 文件中。

### 2.3 查看结果

实验完成后，可以查看 `results.csv` 文件获取所有 39 个 loss 值，用于后续的 Scaling Law 分析。同时，你也可以在 W&B 的项目页面上查看详细的训练曲线和指标。