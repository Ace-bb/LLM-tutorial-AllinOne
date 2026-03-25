"""
DPO 训练脚本

使用 DPO 算法微调语言模型
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import copy

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.data.preprocessing import create_dataloader
from src.models.policy_model import PolicyModel, create_policy_model
from src.algorithms.dpo import DPOTrainer

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


def train_dpo(
    config_path: str = 'config.yaml',
    output_dir: Optional[str] = None
):
    """
    使用 DPO 算法训练模型
    
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
    output_dir = Path(config['training']['output_dir']) / 'dpo'
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
    
    # 创建参考模型（初始策略的副本，固定）
    ref_model = create_policy_model(
        model_name=config['model']['base_model'],
        pad_token_id=config['model']['pad_token_id']
    )
    
    # 如果配置了参考模型路径，加载预训练的参考模型
    if config['dpo'].get('ref_model_path'):
        logger.info(f"加载参考模型：{config['dpo']['ref_model_path']}")
        ref_model = create_policy_model(
            model_name=config['dpo']['ref_model_path'],
            pad_token_id=config['model']['pad_token_id']
        )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config['dpo'].get('learning_rate', 5e-7),
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    # 创建 DPO 训练器
    dpo_trainer = DPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        optimizer=optimizer,
        config=config['dpo'],
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
    
    # 验证数据加载器
    val_loader = None
    if config['data'].get('val_data_path'):
        val_loader = create_dataloader(
            data_path=config['data']['val_data_path'],
            tokenizer=tokenizer,
            batch_size=config['data']['batch_size'],
            max_length=config['model']['max_length'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 0)
        )
    
    # 训练循环
    num_epochs = config['training'].get('num_epochs', 3)
    logging_steps = config['training'].get('logging_steps', 50)
    save_steps = config['training'].get('save_steps', 500)
    
    logger.info(f"开始 DPO 训练，{num_epochs} 个 epochs")
    
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # DPO 训练
        train_metrics = dpo_trainer.train_epoch(
            dataloader=train_loader,
            epoch=epoch + 1,
            logging_steps=logging_steps
        )
        
        logger.info(f"训练完成：{train_metrics}")
        
        # 验证
        if val_loader:
            val_metrics = dpo_trainer.evaluate(val_loader)
            logger.info(f"验证结果：{val_metrics}")
            
            # 保存最佳模型
            if val_metrics['val_accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['val_accuracy']
                save_path = output_dir / 'best_model'
                policy_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info(f"保存最佳模型到 {save_path}")
        
        # 定期保存
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
    
    logger.info("DPO 训练完成!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DPO 训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    train_dpo(
        config_path=args.config,
        output_dir=args.output
    )
