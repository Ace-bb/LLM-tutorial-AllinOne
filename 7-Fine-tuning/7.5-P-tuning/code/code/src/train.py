# -*- coding: utf-8 -*-
"""
P-tuning 训练脚本

完整的训练循环，包括：
- 数据加载和预处理
- 模型初始化
- 训练循环
- 验证和评估
- 模型保存
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, WarmupLR
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer

from .ptuning_model import PTuningModel
from ..config import PTuningConfig, default_config


class TextClassificationDataset(Dataset):
    """
    文本分类数据集
    
    支持自定义文本和标签数据
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128
    ):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表（整数）
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class Trainer:
    """
    P-tuning 训练器
    
    封装完整的训练流程
    """
    
    def __init__(
        self,
        model: PTuningModel,
        tokenizer: AutoTokenizer,
        config: PTuningConfig,
        output_dir: str = "./output"
    ):
        """
        初始化训练器
        
        Args:
            model: PTuningModel 实例
            tokenizer: HuggingFace tokenizer
            config: P-tuning 配置
            output_dir: 输出目录
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 将模型移到设备
        self.model.to(self.device)
        
        # 初始化优化器（仅优化可训练参数）
        self.optimizer = AdamW(
            model.get_trainable_parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 日志记录
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
        }
    
    def prepare_dataloaders(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int]
    ) -> tuple:
        """
        准备训练和验证数据加载器
        
        Args:
            train_texts: 训练文本
            train_labels: 训练标签
            val_texts: 验证文本
            val_labels: 验证标签
        
        Returns:
            train_loader, val_loader
        """
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_seq_length
        )
        val_dataset = TextClassificationDataset(
            val_texts, val_labels, self.tokenizer, self.config.max_seq_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        训练一个 epoch
        
        Args:
            train_loader: 训练数据加载器
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        在验证集上评估
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            评估指标字典
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(val_loader, desc="Evaluating")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 收集预测
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # 计算准确率
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': accuracy
        }
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int]
    ) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Args:
            train_texts: 训练文本
            train_labels: 训练标签
            val_texts: 验证文本
            val_labels: 验证标签
        
        Returns:
            训练历史记录
        """
        print(f"\n开始训练...")
        print(f"  设备：{self.device}")
        print(f"  训练样本：{len(train_texts)}")
        print(f"  验证样本：{len(val_texts)}")
        print(f"  虚拟词元数量：{self.config.num_virtual_tokens}")
        print(f"  学习率：{self.config.learning_rate}")
        print(f"  批大小：{self.config.batch_size}")
        print(f"  训练轮数：{self.config.num_epochs}")
        
        # 打印参数统计
        print(f"\n参数统计:")
        self.model.print_trainable_parameters()
        
        # 准备数据加载器
        train_loader, val_loader = self.prepare_dataloaders(
            train_texts, train_labels, val_texts, val_labels
        )
        
        best_val_accuracy = 0.0
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*50}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            
            print(f"\n训练损失：{train_loss:.4f}")
            print(f"验证损失：{val_metrics['loss']:.4f}")
            print(f"验证准确率：{val_metrics['accuracy']:.4f}")
            
            # 保存最佳模型
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                self.save_checkpoint(f"best_model")
                print(f"✓ 保存最佳模型（准确率：{best_val_accuracy:.4f}）")
            
            # 定期保存
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")
        
        print(f"\n{'='*50}")
        print(f"训练完成！")
        print(f"最佳验证准确率：{best_val_accuracy:.4f}")
        print(f"{'='*50}")
        
        # 保存训练历史
        self.save_training_history()
        
        return self.training_history
    
    def save_checkpoint(self, name: str = "checkpoint"):
        """
        保存检查点
        
        Args:
            name: 检查点名称
        """
        save_path = os.path.join(self.output_dir, name)
        self.model.save_pretrained(save_path)
        print(f"模型已保存到：{save_path}")
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"训练历史已保存到：{history_path}")


def create_sample_data() -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    创建示例数据（用于测试）
    
    Returns:
        train_texts, train_labels, val_texts, val_labels
    """
    # 简单的情感分类示例
    train_texts = [
        "这部电影太棒了，我非常喜欢！",
        "非常糟糕的体验，再也不会来了。",
        "演员表演精彩，剧情紧凑。",
        "浪费时间，电影很无聊。",
        "强烈推荐，年度最佳电影！",
        "失望透顶，完全不推荐。",
        "视觉效果震撼，音效出色。",
        "故事老套，没有新意。",
    ] * 10  # 重复以增加数据量
    
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0] * 10  # 1: 正面，0: 负面
    
    val_texts = [
        "不错的电影，值得一看。",
        "一般般，没有特别的感觉。",
        "太烂了，中途就走了。",
        "超出预期，非常惊喜！",
    ]
    
    val_labels = [1, 0, 0, 1]
    
    return train_texts, train_labels, val_texts, val_labels


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="P-tuning 训练脚本")
    parser.add_argument("--model_name", type=str, default="gpt2", help="预训练模型名称")
    parser.add_argument("--num_virtual_tokens", type=int, default=50, help="虚拟词元数量")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--use_sample_data", action="store_true", help="使用示例数据测试")
    
    args = parser.parse_args()
    
    # 创建配置
    config = P TuningConfig(
        model_name=args.model_name,
        num_virtual_tokens=args.num_virtual_tokens,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )
    
    # 加载 tokenizer
    print(f"加载 tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 设置 pad_token（GPT-2 需要）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建模型
    print(f"\n创建 P-tuning 模型...")
    model = PTuningModel(
        model_name=config.model_name,
        num_virtual_tokens=config.num_virtual_tokens,
        encoder_hidden_dim=config.encoder_hidden_dim,
        num_labels=config.num_labels,
        task_type=config.task_type,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir=args.output_dir
    )
    
    # 准备数据
    if args.use_sample_data:
        print("\n使用示例数据进行测试...")
        train_texts, train_labels, val_texts, val_labels = create_sample_data()
    else:
        # TODO: 加载真实数据集
        print("\n请实现数据加载逻辑...")
        print("示例：使用 HuggingFace datasets 库加载 IMDB 数据集")
        return
    
    # 开始训练
    trainer.train(train_texts, train_labels, val_texts, val_labels)
    
    print(f"\n训练完成！模型保存在：{args.output_dir}")


if __name__ == "__main__":
    main()
