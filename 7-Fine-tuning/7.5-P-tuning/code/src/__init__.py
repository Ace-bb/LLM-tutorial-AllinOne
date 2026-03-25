# -*- coding: utf-8 -*-
"""
P-tuning 源代码模块

包含：
- prompt_encoder: LSTM+MLP 提示编码器
- ptuning_model: P-tuning 模型封装
- train: 训练脚本
- inference: 推理脚本
"""

from .prompt_encoder import PromptEncoder, SimplePromptEncoder
from .ptuning_model import PTuningModel

__all__ = [
    "PromptEncoder",
    "SimplePromptEncoder",
    "PTuningModel",
]

__version__ = "1.0.0"
