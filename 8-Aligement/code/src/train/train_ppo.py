"""
PPO 训练脚本

使用 PPO 算法微调语言模型
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.data.preprocessing import create_dataloader
from src.models.policy_model import PolicyModel, ValueModel, create_policy_model, create_value_model
from src.algorithms.ppo import PPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_ppo(
    config_path: str = 'config.yaml',
    output_dir: Optional[str] = None
):
    """
    使用 PPO 算法训练模型
    
    Args:
        config_path: 配置文件路径
        output_dir: 输出目录（可选，覆盖配置）
    """
    # 加载配置
    config = load_config(config_path)
    
    if output_dir:
        config['training']['output_dir'] = output_dir
    
    # 设置设备
    device = config['model'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，使用 CPU")
        device = 'cpu'
    
    logger.info(f"使用设备：{device}")
    
    # 创建输出目录
    output_dir = Path(config['training']['output_dir']) / 'ppo'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config['model']['pad_token_id'] = tokenizer.eos_token_id
    
    # 创建策略模型
    policy_model = create_policy_model(
        model_name=config['model']['base_model'],
        pad_token_id=config['model']['pad_token_id']
    )
    
    # 创建价值模型
    value_model = create_value_model(
        model_name=config['model']['base_model'],
        hidden_dim=256,
        dropout=0.1
    )
    
    # 创建优化器
    optimizer_policy = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config['ppo'].get('learning_rate', 1e-5),
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    optimizer_value = torch.optim.AdamW(
        value_model.parameters(),
        lr=config['ppo'].get('learning_rate', 1e-5),
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    # 创建 PPO 训练器
    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        value_model=value_model,
        optimizer_policy=optimizer_policy,
        optimizer_value=optimizer_value,
        config=config['ppo'],
        device=device
    )
    
    # 创建数据加载器
    train_loader = create_dataloader(
        data_path=config['data']['train_data_path'],
        tokenizer=tokenizer,
        batch_size=config['data']['batch_size'],
        max_length=config['model']['max_length'],
        shuffle=config['data']['shuffle'],
        num_workers=config['data'].get('num_workers', 0)
    )
    
    # 训练循环
    num_epochs = config['training'].get('num_epochs', 3)
    logging_steps = config['training'].get('logging_steps', 50)
    save_steps = config['training'].get('save_steps', 500)
    
    logger.info(f"开始 PPO 训练，{num_epochs} 个 epochs")
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # PPO 训练
        # 注意：PPO 需要特殊的 rollout 数据收集
        # 这里简化示例，实际使用需要：
        # 1. 使用当前策略生成响应
        # 2. 计算奖励（使用 reward model 或规则）
        # 3. 计算 GAE 优势
        # 4. 进行 PPO 更新
        
        train_metrics = ppo_trainer.train_epoch(
            dataloader=train_loader,
            epoch=epoch + 1,
            logging_steps=logging_steps
        )
        
        logger.info(f"训练完成：{train_metrics}")
        
        # 保存检查点
        if (epoch + 1) % (save_steps // len(train_loader)) == 0:
            save_path = output_dir / f'checkpoint_epoch_{epoch + 1}'
            policy_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"保存检查点到 {save_path}")
    
    # 保存最终模型
    final_path = output_dir / 'final_model'
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"保存最终模型到 {final_path}")
    
    logger.info("PPO 训练完成!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO 训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    train_ppo(
        config_path=args.config,
        output_dir=args.output
    )
