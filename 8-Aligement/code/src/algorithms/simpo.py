"""
SimPO (Simple Preference Optimization) 实现

简单偏好优化算法，无需参考模型
使用长度归一化的平均 log 概率，简化训练并提高性能

核心思想:
直接使用平均 log 概率作为分数，无需参考模型
通过 target margin γ 控制 chosen 和 rejected 的差距

核心公式:
1. 长度归一化的平均 log 概率:
   score(x, y) = (1/|y|) · Σ log P(y_i | x, y_{<i})

2. SimPO 损失:
   L_SimPO = -E[log(σ(β · (score(x, y_w) - score(x, y_l)) - γ))]
   
   其中:
   - y_w: chosen response
   - y_l: rejected response
   - β: 温度参数
   - γ: target margin (期望的分数差距)

优势:
- 无需参考模型，降低计算成本
- 长度归一化避免偏好长文本的偏差
- 简单高效，性能优异
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def compute_simpo_loss(
    chosen_log_probs: torch.Tensor,
    rejected_log_probs: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
    gamma: float = 0.5,
    use_length_norm: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算 SimPO 损失
    
    SimPO 使用长度归一化的平均 log 概率
    
    Args:
        chosen_log_probs: chosen 序列的 token log 概率 [batch_size, seq_len]
        rejected_log_probs: rejected 序列的 token log 概率 [batch_size, seq_len]
        chosen_labels: chosen 序列的 labels
        rejected_labels: rejected 序列的 labels
        beta: 温度参数 (默认 0.1)
        gamma: target margin (默认 0.5)
        use_length_norm: 是否使用长度归一化 (默认 True)
    
    Returns:
        loss: SimPO 损失 (标量)
        metrics: 指标字典
    """
    # ===== 计算长度归一化的平均 log 概率 =====
    if use_length_norm:
        chosen_scores = average_log_probs(chosen_log_probs, chosen_labels)
        rejected_scores = average_log_probs(rejected_log_probs, rejected_labels)
    else:
        # 不使用长度归一化，直接求和
        mask_chosen = (chosen_labels != -100) & (chosen_labels != 0)
        mask_rejected = (rejected_labels != -100) & (rejected_labels != 0)
        
        chosen_scores = (chosen_log_probs * mask_chosen.float()).sum(dim=1)
        rejected_scores = (rejected_log_probs * mask_rejected.float()).sum(dim=1)
    
    # ===== 计算 SimPO 损失 =====
    # 分数差异
    score_diff = chosen_scores - rejected_scores
    
    # SimPO 损失：-log(sigmoid(β · (score_diff - γ)))
    # 移项：β · score_diff - β · γ
    logits = beta * score_diff - beta * gamma
    
    losses = -F.logsigmoid(logits)
    loss = losses.mean()
    
    # ===== 计算指标 =====
    with torch.no_grad():
        # 准确率
        predictions = (score_diff > gamma).float()
        accuracy = predictions.mean().item()
        
        # 分数差异统计
        score_margin = score_diff.mean().item()
        score_std = score_diff.std().item()
        
        # 分数统计
        chosen_score_mean = chosen_scores.mean().item()
        rejected_score_mean = rejected_scores.mean().item()
    
    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy,
        'score_margin': score_margin,
        'score_std': score_std,
        'chosen_score_mean': chosen_score_mean,
        'rejected_score_mean': rejected_score_mean,
        'gamma': gamma,
        'beta': beta
    }
    
    return loss, metrics


def average_log_probs(
    log_probs: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    计算长度归一化的平均 log 概率
    
    Args:
        log_probs: token log 概率 [batch_size, seq_len]
        labels: labels [batch_size, seq_len]
    
    Returns:
        平均 log 概率 [batch_size]
    """
    # 创建 mask（排除 padding）
    mask = (labels != -100) & (labels != 0)
    mask = mask.float()
    
    # 计算加权和
    sum_log_probs = (log_probs * mask).sum(dim=1)
    lengths = mask.sum(dim=1)
    
    # 避免除零
    lengths = torch.clamp(lengths, min=1)
    
    avg_log_probs = sum_log_probs / lengths
    
    return avg_log_probs


def get_token_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    获取每个 token 的 log 概率
    
    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]
    
    Returns:
        token_log_probs: [batch_size, seq_len]
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    
    token_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)
    
    return token_log_probs


class SimPOTrainer:
    """
    SimPO 训练器
    
    实现完整的 SimPO 训练流程：
    - 前向传播
    - SimPO 损失计算
    - 反向传播和优化
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        初始化 SimPO 训练器
        
        Args:
            model: 策略模型（可训练）
            optimizer: 优化器
            config: 配置字典，包含：
                - gamma: target margin
                - beta: 温度参数
                - use_length_norm: 是否使用长度归一化
            device: 训练设备
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # 从配置中读取超参数
        self.gamma = config.get('gamma', 0.5)
        self.beta = config.get('beta', 0.1)
        self.use_length_norm = config.get('use_length_norm', True)
        
        logger.info(f"SimPOTrainer 初始化完成，gamma={self.gamma}, beta={self.beta}")
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        单步 SimPO 训练
        
        Args:
            batch: 数据批次，包含：
                - chosen_input_ids, chosen_attention_mask
                - rejected_input_ids, rejected_attention_mask
        
        Returns:
            loss: SimPO 损失
            metrics: 指标字典
        """
        self.model.train()
        
        # 将数据移动到设备
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # ===== 前向传播 =====
        # Chosen
        chosen_outputs = self.model(
            input_ids=batch['chosen_input_ids'],
            attention_mask=batch['chosen_attention_mask']
        )
        
        # Rejected
        rejected_outputs = self.model(
            input_ids=batch['rejected_input_ids'],
            attention_mask=batch['rejected_attention_mask']
        )
        
        # 获取 token log 概率
        chosen_log_probs = get_token_log_probs(
            chosen_outputs.logits,
            batch['chosen_input_ids']
        )
        rejected_log_probs = get_token_log_probs(
            rejected_outputs.logits,
            batch['rejected_input_ids']
        )
        
        # ===== 计算 SimPO 损失 =====
        loss, metrics = compute_simpo_loss(
            chosen_log_probs=chosen_log_probs,
            rejected_log_probs=rejected_log_probs,
            chosen_labels=batch['chosen_input_ids'],
            rejected_labels=batch['rejected_input_ids'],
            beta=self.beta,
            gamma=self.gamma,
            use_length_norm=self.use_length_norm
        )
        
        # ===== 反向传播 =====
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss, metrics
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        logging_steps: int = 50
    ) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            dataloader: 训练数据 DataLoader
            epoch: 当前 epoch
            logging_steps: 日志步数
        
        Returns:
            平均训练指标
        """
        total_loss = 0.0
        total_accuracy = 0.0
        total_score_margin = 0.0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            loss, metrics = self.train_step(batch)
            
            total_loss += metrics.get('loss', loss.item())
            total_accuracy += metrics.get('accuracy', 0.0)
            total_score_margin += metrics.get('score_margin', 0.0)
            num_batches += 1
            
            # 日志
            if (step + 1) % logging_steps == 0:
                avg_loss = total_loss / num_batches
                avg_accuracy = total_accuracy / num_batches
                avg_margin = total_score_margin / num_batches
                logger.info(
                    f"Epoch {epoch}, Step {step + 1}, "
                    f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, "
                    f"Margin: {avg_margin:.4f}"
                )
        
        # 计算平均指标
        return {
            'train_loss': total_loss / num_batches,
            'train_accuracy': total_accuracy / num_batches,
            'train_score_margin': total_score_margin / num_batches
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            dataloader: 验证数据 DataLoader
        
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            chosen_outputs = self.model(
                input_ids=batch['chosen_input_ids'],
                attention_mask=batch['chosen_attention_mask']
            )
            rejected_outputs = self.model(
                input_ids=batch['rejected_input_ids'],
                attention_mask=batch['rejected_attention_mask']
            )
            
            # 获取 token log 概率
            chosen_log_probs = get_token_log_probs(
                chosen_outputs.logits,
                batch['chosen_input_ids']
            )
            rejected_log_probs = get_token_log_probs(
                rejected_outputs.logits,
                batch['rejected_input_ids']
            )
            
            # 计算损失
            loss, metrics = compute_simpo_loss(
                chosen_log_probs=chosen_log_probs,
                rejected_log_probs=rejected_log_probs,
                chosen_labels=batch['chosen_input_ids'],
                rejected_labels=batch['rejected_input_ids'],
                beta=self.beta,
                gamma=self.gamma,
                use_length_norm=self.use_length_norm
            )
            
            total_loss += metrics.get('loss', loss.item())
            total_accuracy += metrics.get('accuracy', 0.0)
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
