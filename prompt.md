角色： 你是一名资深的 AI 研究工程师，擅长使用 PyTorch、Hugging Face Transformers 和 TRL 库进行大规模模型训练。

任务： 请为我开发一个用于研究 SFT (Supervised Fine-Tuning) Scaling Law 的代码项目。项目的核心是训练一个语言模型在不同大小和配比的数据集上，并评估其在固定验证集上的损失（loss）。请遵循以下详细需求，生成清晰、模块化且可直接运行的代码。

项目核心需求：

我们的目标是探索当改变特定领域（如 "math", "code", "science"）的数据量时，模型在这些领域以及其他领域上的性能变化。为此，我们将构建一个数据合成器来创建多个训练数据集，然后使用一个统一的训练和评估流程来获取模型在不同验证集上的 loss。

具体技术和功能要求：

1. 训练框架 (Training Framework)

库： 使用 Hugging Face TRL 的 SFTTrainer。

训练策略：

全量微调 (Full Fine-Tuning)： 不使用 LoRA 或任何 PEFT 技术。

分布式训练： 必须支持 FSDP (Fully Sharded Data Parallelism)，以便在多 GPU 环境下进行高效的全量微调。

性能优化： 集成 Flash Attention 2 以加速训练和减少显存占用。

模型： 使用 Qwen/Qwen3-8B 作为基础模型。

序列长度： max_sequence_length 固定为 8192。

日志记录： 集成 wandb 用于记录训练过程中的指标（如 training loss）。

配置：

所有训练参数（如学习率、批大小、周期数等）和模型配置应通过一个 config.yaml 文件进行管理。

项目的启动应通过一个 .sh 脚本完成，该脚本负责调用数据处理、训练和评估流程。

2. 数据合成器 (Data Synthesizer)

源数据： 使用 Hugging Face Hub 上的 mixture-of-thoughts 数据集，并专注于其三个子集："math", "code", "science"。

基础训练数据池：

从 "math" 子集采样，直到累计 token 数量恰好超过 660,000 个，形成数据集 D1。

从 "code" 子集采样，直到累计 token 数量恰好超过 660,000 个，形成数据集 D2。

从 "science" 子集采样，直到累计 token 数量恰好超过 660,000 个，形成数据集 D3。

注意： 采样时应是完整的样本（example），不能截断。你需要迭代数据集，对每个样本进行分词（tokenize）并累加其 token 数量，直到满足条件。

生成13个训练变体：

基线数据集 (Baseline)： D_base = D1 + D2 + D3。

扰动 D1 (math)： 保持 D2 和 D3 不变，对 D1 进行以下四种变换，生成四个新的训练集：

1/3 * D1 + D2 + D3  (从 D1 中随机采样 1/3 的样本)

1/2 * D1 + D2 + D3  (从 D1 中随机采样 1/2 的样本)

2 * D1 + D2 + D3    (将 D1 重复 2 次)

3 * D1 + D2 + D3    (将 D1 重复 3 次)

扰动 D2 (code)： 保持 D1 和 D3 不变，对 D2 进行与上述相同的四种变换，生成四个新的训练集。

扰动 D3 (science)： 保持 D1 和 D2 不变，对 D3 进行与上述相同的四种变换，生成四个新的训练集。

数据处理： 数据合成器应将生成的 13 个 Dataset 对象保存到磁盘（例如，使用 dataset.save_to_disk），以便训练脚本可以重复使用，避免每次重新生成。

3. 验证集 (Validation Sets)

独立于训练数据，创建三个验证集：

V1 (math): 从 "math" 子集采样，直到累计 token 数量恰好超过 1,000,000 个。

V2 (code): 从 "code" 子集采样，直到累计 token 数量恰好超过 1,000,000 个。

V3 (science): 从 "science" 子集采样，直到累计 token 数量恰好超过 1,000,000 个。

同样，将这三个验证集保存到磁盘。

4. 评估流程 (Evaluation Process)

对于上述生成的 13 个训练数据集中的每一个，执行以下操作：

使用 SFTTrainer 进行模型训练。

训练完成后，加载训练好的模型 checkpoint。

分别在 V1, V2, V3 三个验证集上进行评估。

评估指标为 样本级别的平均 loss (sample-level average loss)。你需要手动编写评估逻辑：遍历验证集中的每一个样本，通过模型进行前向传播计算其 loss，然后计算所有样本 loss 的平均值。

结果记录：

最终需要记录 13 * 3 = 39 个 loss 值。

请将结果输出到一个结构化的文件（如 results.csv 或 results.json）中，清晰地标明每个 loss 值对应的训练数据集和验证数据集。例如：train_dataset_name, validation_dataset_name, loss。

代码实现要求：

代码风格： 请编写简洁、高质量、模块化的 Python 代码。添加适当的注释来解释关键部分。

简洁性： 严格按照需求实现，不添加不必要的功能或模块（例如，暂时不需要 try-except 错误处理）。

文件结构： 请遵循下面我提供的推荐文件结构来组织代码。
sft_trainer/
├── configs/
│   └── training_config.yaml      # 训练参数、模型名称、W&B 配置等
│
├── data/                         # 此目录用于存放生成的数据集，.gitignore应包含此目录
│   ├── training/                 # 存放13个训练集
│   └── validation/               # 存放3个验证集
│
├── scripts/
│   └── run_experiments.sh        # 自动化整个流程的主脚本
│
├── src/
│   ├── __init__.py
│   ├── data_builder.py           # 数据合成器，用于生成和保存13个训练集和3个验证集
│   ├── train.py                  # SFT训练脚本，接收数据集路径作为参数
│   └── evaluate.py               # 评估脚本，计算并输出在指定验证集上的loss
│
├── requirements.txt              # 项目依赖
└── README.md                     # 项目说明、环境配置和使用指南