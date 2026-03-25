# -*- coding: utf-8 -*-
"""
P-tuning 超参数配置文件

包含所有可配置的超参数，便于实验调整和复现。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PTuningConfig:
    """P-tuning 配置类"""
    
    # ========== 模型配置 ==========
    model_name: str = "gpt2"  # 预训练模型名称
    num_virtual_tokens: int = 50  # 虚拟词元数量（20-100，根据任务复杂度调整）
    
    # ========== 编码器配置 ==========
    encoder_hidden_dim: int = 512  # LSTM 隐藏层维度
    encoder_num_layers: int = 1  # LSTM 层数
    encoder_bidirectional: bool = True  # 是否使用双向 LSTM
    
    # ========== 训练配置 ==========
    learning_rate: float = 1e-3  # 学习率（比全量微调稍大）
    weight_decay: float = 0.01  # 权重衰减
    batch_size: int = 16  # 批大小
    num_epochs: int = 100  # 训练轮数
    max_seq_length: int = 128  # 最大序列长度
    
    # ========== 优化器配置 ==========
    optimizer: str = "adamw"  # 优化器类型
    scheduler: str = "linear"  # 学习率调度器
    warmup_ratio: float = 0.1  # 预热比例
    
    # ========== 其他配置 ==========
    seed: int = 42  # 随机种子
    device: str = "cuda"  # 训练设备
    logging_steps: int = 10  # 日志记录步数
    save_steps: int = 100  # 模型保存步数
    
    # ========== 任务配置 ==========
    task_type: str = "classification"  # 任务类型：classification, generation
    num_labels: int = 2  # 分类标签数量（仅分类任务）
    
    def __post_init__(self):
        """验证配置参数"""
        if self.num_virtual_tokens <= 0:
            raise ValueError("虚拟词元数量必须大于 0")
        if self.encoder_hidden_dim <= 0:
            raise ValueError("LSTM 隐藏层维度必须大于 0")
        if self.learning_rate <= 0:
            raise ValueError("学习率必须大于 0")


# 默认配置实例
default_config = PTuningConfig()


# ========== 不同任务的推荐配置 ==========

TASK_CONFIGS = {
    # 情感分类（简单任务）
    "sentiment_classification": {
        "num_virtual_tokens": 30,
        "num_epochs": 50,
        "num_labels": 2,
    },
    
    # 自然语言推理（复杂任务）
    "nli": {
        "num_virtual_tokens": 80,
        "num_epochs": 100,
        "num_labels": 3,
    },
    
    # 文本生成任务
    "generation": {
        "num_virtual_tokens": 50,
        "num_epochs": 100,
        "task_type": "generation",
    },
    
    # 小样本学习（需要更多轮数）
    "few_shot": {
        "num_virtual_tokens": 100,
        "num_epochs": 200,
        "learning_rate": 5e-4,
    },
}


def get_task_config(task_name: str, base_config: Optional['PTuningConfig'] = None) -> 'PTuningConfig':
    """
    获取特定任务的推荐配置
    
    Args:
        task_name: 任务名称
        base_config: 基础配置，如为 None 则使用默认配置
    
    Returns:
        配置好的 PTuningConfig 实例
    """
    if base_config is None:
        config = PTuningConfig()
    else:
        config = base_config
    
    if task_name in TASK_CONFIGS:
        task_cfg = TASK_CONFIGS[task_name]
        for key, value in task_cfg.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config


if __name__ == "__main__":
    # 测试配置
    print("默认配置:")
    print(f"  虚拟词元数量：{default_config.num_virtual_tokens}")
    print(f"  LSTM 隐藏层：{default_config.encoder_hidden_dim}")
    print(f"  学习率：{default_config.learning_rate}")
    print(f"  批大小：{default_config.batch_size}")
    
    print("\n情感分类任务配置:")
    sentiment_config = get_task_config("sentiment_classification")
    print(f"  虚拟词元数量：{sentiment_config.num_virtual_tokens}")
    print(f"  训练轮数：{sentiment_config.num_epochs}")
