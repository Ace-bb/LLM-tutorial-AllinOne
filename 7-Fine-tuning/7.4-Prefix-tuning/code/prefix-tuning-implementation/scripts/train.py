#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prefix-tuning 训练脚本

使用示例：
    # 训练 GPT-2 进行文本生成
    python scripts/train.py \
        --model_name gpt2 \
        --model_type causal \
        --prefix_length 20 \
        --output_dir ./output/gpt2-prefix \
        --num_train_epochs 10

    # 训练 BART 进行序列到序列任务
    python scripts/train.py \
        --model_name facebook/bart-base \
        --model_type seq2seq \
        --prefix_length 30 \
        --output_dir ./output/bart-prefix \
        --num_train_epochs 15
"""

import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import yaml
import os
import sys

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prefix_model import PrefixTuningModel, PrefixTuningConfig
from trainer import PrefixTrainer, TrainingArguments


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Prefix-tuning 训练脚本")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="预训练模型名称或路径")
    parser.add_argument("--model_type", type=str, default="causal",
                        choices=["causal", "seq2seq"],
                        help="模型类型：causal (GPT-2) 或 seq2seq (BART)")
    parser.add_argument("--prefix_length", type=int, default=20,
                        help="前缀长度")
    parser.add_argument("--bottleneck_size", type=int, default=512,
                        help="重参数化瓶颈层大小")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="学习率")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="每设备训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热比例")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="最大梯度范数")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存检查点步数")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="评估步数")
    parser.add_argument("--fp16", action="store_true",
                        help="是否使用混合精度训练")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # 数据集参数
    parser.add_argument("--dataset_name", type=str, default="squad",
                        help="数据集名称")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="最大序列长度")
    
    return parser.parse_args()


def load_training_data(dataset_name: str, tokenizer, max_seq_length: int = 512):
    """
    加载和预处理训练数据
    
    Args:
        dataset_name: 数据集名称
        tokenizer: Tokenizer
        max_seq_length: 最大序列长度
    
    Returns:
        训练数据集和评估数据集
    """
    print(f"加载数据集：{dataset_name}")
    
    # 这里以 SQuAD 为例，实际使用时可替换为其他数据集
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"加载数据集失败：{e}")
        print("使用示例数据...")
        # 创建示例数据
        from torch.utils.data import TensorDataset
        import torch
        
        # 创建随机数据用于演示
        input_ids = torch.randint(0, 1000, (100, max_seq_length))
        attention_mask = torch.ones(100, max_seq_length)
        labels = input_ids.clone()
        
        train_dataset = TensorDataset(input_ids[:80], attention_mask[:80], labels[:80])
        eval_dataset = TensorDataset(input_ids[80:], attention_mask[80:], labels[80:])
        return train_dataset, eval_dataset
    
    # 预处理函数
    def preprocess_function(examples):
        # 根据具体数据集调整预处理逻辑
        texts = examples.get('question', []) + examples.get('context', [''] * len(examples.get('question', [])))
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # 应用预处理
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset.get("validation", tokenized_dataset["train"])
    
    return train_dataset, eval_dataset


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建配置
    config = PrefixTuningConfig(
        model_name_or_path=args.model_name,
        model_type=args.model_type,
        prefix_length=args.prefix_length,
        bottleneck_size=args.bottleneck_size,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"使用设备：{config.device}")
    print(f"模型：{config.model_name_or_path}")
    print(f"前缀长度：{config.prefix_length}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据
    train_dataset, eval_dataset = load_training_data(
        args.dataset_name,
        tokenizer,
        args.max_seq_length
    )
    print(f"训练样本数：{len(train_dataset)}")
    print(f"评估样本数：{len(eval_dataset)}")
    
    # 创建 Prefix-tuning 模型
    print("创建 Prefix-tuning 模型...")
    model = PrefixTuningModel(config)
    model.to(config.device)
    
    # 打印可训练参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数：{total_params:,}")
    print(f"可训练参数数：{trainable_params:,}")
    print(f"可训练参数比例：{trainable_params / total_params * 100:.2f}%")
    
    # 创建训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        fp16=args.fp16,
        seed=args.seed
    )
    
    # 创建 Trainer
    trainer = PrefixTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    print("保存最终模型...")
    trainer.save_model()
    
    print("训练完成！")


if __name__ == "__main__":
    main()
