"""
ORPO (Odds Ratio Policy Optimization) 实现

Odds Ratio 策略优化，无需参考模型
结合 SFT 损失和 Odds Ratio 损失，简化训练流程

核心思想:
使用 odds ratio 来衡量模型对 chosen 和 rejected 的偏好程度
无需单独的参考模型，降低计算成本

核心公式:
1. Odds Ratio:
   OR(x, y_w, y_l) = odds(y_w|x) / odds(y_l|x)
   
   其中 odds(y|x) = P(y|x) / (1 - P(y|x))

2. OR 损失:
   L_OR = -log(σ(log(OR(x, y_w, y_l))))
        = -log(σ(log(odds(y_w|x)) - log(odds(y_l|x))))

3. 总损失:
   L = L_SFT + λ · L_OR
   
   其中 L_SFT 是标准的监督微调损失
   λ 控制 OR 损失的权重
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def compute_orpo_loss(
    chosen_logits: torch.Tensor,
    rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
    lambda_or: float = 0.5,
    include_sft_loss: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算 ORPO 损失
    
    ORPO 结合 SFT 损失和 Odds Ratio 损失
    
    Args:
        chosen_logits: chosen 序列的 logits [batch_size, seq_len, vocab_size]
        rejected_logits: rejected 序列的 logits [batch_size, seq_len, vocab_size]
        chosen_labels: chosen 序列的 labels
        rejected_labels: rejected 序列的 labels
        beta: 温度参数
        lambda_or: OR 损失权重
        include_sft_loss: 是否包含 SFT 损失
    
    Returns:
        total_loss: 总损失 (标量)
        metrics: 指标字典
    """
    batch_size = chosen_logits.size(0)
    
    # ===== 计算 SFT 损失 =====
    sft_loss = torch.tensor(0.0, device=chosen_logits.device)
    if include_sft_loss:
        # Chosen 的 SFT 损失
        sft_loss_chosen = compute_sft_loss(chosen_logits, chosen_labels)
        # Rejected 的 SFT 损失
        sft_loss_rejected = compute_sft_loss(rejected_logits, rejected_labels)
        # 平均 SFT 损失
        sft_loss = (sft_loss_chosen + sft_loss_rejected) / 2
    
    # ===== 计算 Odds Ratio 损失 =====
    # 计算 chosen 和 rejected 的 log 概率
    chosen_log_probs = get_token_log_probs(chosen_logits, chosen_labels)
    rejected_log_probs = get_token_log_probs(rejected_logits, rejected_labels)
    
    # 计算平均 log 概率（长度归一化）
    chosen_avg_log_probs = average_log_probs(chosen_log_probs, chosen_labels)
    rejected_avg_log_probs = average_log_probs(rejected_log_probs, rejected_labels)
    
    # 计算 odds: odds = P / (1 - P)
    # 使用 log odds: log(odds) = log(P) - log(1 - P)
    # 近似：log(odds) ≈ log(P) （当 P 接近 1 时）
    # 更准确：使用 logit 函数
    
    # 计算 log odds ratio
    # log(OR) = log(odds_chosen) - log(odds_rejected)
    # 简化为 log 概率差异
    log_odds_ratio = beta * (chosen_avg_log_probs - rejected_avg_log_probs)
    
    # OR 损失：-log(sigmoid(log_odds_ratio))
    or_loss = -F.logsigmoid(log_odds_ratio).mean()
    
    # ===== 总损失 =====
    total_loss = sft_loss + lambda_or * or_loss
    
    # ===== 计算指标 =====
    with torch.no_grad():
        # 准确率
        predictions = (log_odds_ratio > 0).float()
        accuracy = predictions.mean().item()
        
        # Odds ratio 统计
        or_margin = log_odds_ratio.mean().item()
        
        # SFT 损失
        sft_loss_val = sft_loss.item()
        or_loss_val = or_loss.item()
    
    metrics = {
        'total_loss': total_loss.item(),
        'sft_loss': sft_loss_val,
        'or_loss': or_loss_val,
        'accuracy': accuracy,
        'or_margin': or_margin,
        'chosen_log_probs_mean': chosen_avg_log_probs.mean().item(),
        'rejected_log_probs_mean': rejected_avg_log_probs.mean().item()
    }
    
    return total_loss, metrics


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    计算标准 SFT 损失（交叉熵）
    
    Args:
        logits: 模型输出 logits [batch_size, seq_len, vocab_size]
        labels: 标签 token IDs [batch_size, seq_len]
    
    Returns:
        平均交叉熵损失
    """
    # 移位：预测下一个 token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # 计算交叉熵
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss


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


def average_log_probs(
    log_probs: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    计算平均 log 概率（长度归一化）
    
    Args:
        log_probs: [batch_size, seq_len]
        labels: [batch_size, seq_len]
    
    Returns:
        avg_log_probs: [batch_size]
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


class ORPOTrainer:
    """
    ORPO 训练器
    
    实现完整的 ORPO 训练流程：
    - 前向传播
    - SFT + OR 损失计算
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
        初始化 ORPO 训练器
        
        Args:
            model: 策略模型（可训练）
            optimizer: 优化器
            config: 配置字典，包含：
                - lambda: OR 损失权重
                - beta: 温度参数
                - include_sft_loss: 是否包含 SFT 损失
            device: 训练设备
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # 从配置中读取超参数
        self.lambda_or = config.get('lambda', 0.5)
        self.beta = config.get('beta', 0.1)
        self.include_sft_loss = config.get('include_sft_loss', True)
        
        logger.info(f"ORPOTrainer 初始化完成，lambda={self.lambda_or}, beta={self.beta}")
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        单步 ORPO 训练
        
        Args:
            batch: 数据批次，包含：
                - chosen_input_ids, chosen_attention_mask
                - rejected_input_ids, rejected_attention_mask
        
        Returns:
            loss: 总损失
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
        
        # ===== 计算损失 =====
        loss, metrics = compute_orpo_loss(
            chosen_logits=chosen_outputs.logits,
            rejected_logits=rejected_outputs.logits,
            chosen_labels=batch['chosen_input_ids'],
            rejected_labels=batch['rejected_input_ids'],
            beta=self.beta,
            lambda_or=self.lambda_or,
            include_sft_loss=self.include_sft_loss
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
        total_sft_loss = 0.0
        total_or_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            loss, metrics = self.train_step(batch)
            
            total_loss += metrics.get('total_loss', loss.item())
            total_sft_loss += metrics.get('sft_loss', 0.0)
            total_or_loss += metrics.get('or_loss', 0.0)
            total_accuracy += metrics.get('accuracy', 0.0)
            num_batches += 1
            
            # 日志
            if (step + 1) % logging_steps == 0:
                avg_loss = total_loss / num_batches
                avg_accuracy = total_accuracy / num_batches
                logger.info(
                    f"Epoch {epoch}, Step {step + 1}, "
                    f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}"
                )
        
        # 计算平均指标
        return {
            'train_loss': total_loss / num_batches,
            'train_sft_loss': total_sft_loss / num_batches,
            'train_or_loss': total_or_loss / num_batches,
            'train_accuracy': total_accuracy / num_batches
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
            
            # 计算损失
            loss, metrics = compute_orpo_loss(
                chosen_logits=chosen_outputs.logits,
                rejected_logits=rejected_outputs.logits,
                chosen_labels=batch['chosen_input_ids'],
                rejected_labels=batch['rejected_input_ids'],
                beta=self.beta,
                lambda_or=self.lambda_or,
                include_sft_loss=self.include_sft_loss
            )
            
            total_loss += metrics.get('total_loss', loss.item())
            total_accuracy += metrics.get('accuracy', 0.0)
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
