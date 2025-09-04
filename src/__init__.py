#!/usr/bin/env python3
"""
SFT Scaling Law 研究项目

本模块包含用于研究监督微调(SFT) Scaling Law的核心组件：
- data_builder: 数据合成器，用于生成训练和验证数据集
- train: SFT训练脚本，使用TRL的SFTTrainer
- evaluate: 评估脚本，计算样本级别的平均loss
"""

__version__ = "1.0.0"
__author__ = "SFT Scaling Law Research Team"
__description__ = "SFT Scaling Law研究项目核心模块"

# 导入主要组件
from .data_builder import DataBuilder
from .train import SFTTrainingPipeline
from .evaluate import ModelEvaluator

__all__ = [
    "DataBuilder",
    "SFTTrainingPipeline", 
    "ModelEvaluator"
]