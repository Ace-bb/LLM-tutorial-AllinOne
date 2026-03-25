"""
DPO (Direct Preference Optimization) 实现

直接偏好优化算法，无需显式奖励模型
直接优化偏好损失，简化训练流程

核心思想:
将奖励模型隐式地参数化为策略模型和参考模型的差异

核心公式:
1. 隐式奖励:
   r(x, y) = β · log(π(y|x) / π_ref(y|x)) + β · log(Z(x))

2. DPO 损失:
   L_DPO = -E[log(σ(β · log(π(y_w|x)/π_ref(y_w|x)) - β · log(π(y_l|x)/π_ref(y_l|x))))]
   
   简化为:
   L_DPO = -E[log(σ(β · (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)))))]

其中:
- y_w: chosen response (更好的回复)
- y_l: rejected response (较差的回复)
- π: 策略模型
- π_ref: 参考模型（固定）
- β: 温度参数，控制 KL 散度惩罚强度
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def compute_dpo_loss(
    policy_chosen_log_probs: torch.Tensor,
    policy_rejected_log_probs: torch.Tensor,
    ref_chosen_log_probs: torch.Tensor,
    ref_rejected_log_probs: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算 DPO 损失
    
    DPO 损失基于 Bradley-Terry 模型，直接优化偏好
    
    Args:
        policy_chosen_log_probs: 策略模型对 chosen 的 log 概率 [batch_size]
        policy_rejected_log_probs: 策略模型对 rejected 的 log 概率 [batch_size]
        ref_chosen_log_probs: 参考模型对 chosen 的 log 概率 [batch_size]
        ref_rejected_log_probs: 参考模型对 rejected 的 log 概率 [batch_size]
        beta: 温度参数，控制 KL 散度惩罚 (默认 0.1)
        label_smoothing: 标签平滑 (默认 0.0)
    
    Returns:
        loss: DPO 损失 (标量)
        metrics: 指标字典
    """
    # 计算策略模型和参考模型的 log 概率差异
    # 这相当于隐式奖励的差异
    policy_log_ratio = policy_chosen_log_probs - policy_rejected_log_probs
    ref_log_ratio = ref_chosen_log_probs - ref_rejected_log_probs
    
    # 计算隐式奖励差异
    # β · (log(π_w/π_ref_w) - log(π_l/π_ref_l))
    # = β · ((log π_w - log π_ref_w) - (log π_l - log π_ref_l))
    # = β · ((log π_w - log π_l) - (log π_ref_w - log π_ref_l))
    implicit_rewards = beta * (policy_log_ratio - ref_log_ratio)
    
    # DPO 损失：负对数似然
    # L = -log(σ(implicit_rewards))
    # 使用 log_sigmoid 保证数值稳定性
    losses = -F.logsigmoid(implicit_rewards)
    
    # 标签平滑（可选）
    if label_smoothing > 0:
        # 平滑版本：部分质量分配给错误标签
        losses = (
            (1 - label_smoothing) * losses
            + label_smoothing * (-F.logsigmoid(-implicit_rewards))
        )
    
    # 平均损失
    loss = losses.mean()
    
    # 计算指标
    with torch.no_grad():
        # 准确率：预测正确的比例
        predictions = (implicit_rewards > 0).float()
        accuracy = predictions.mean().item()
        
        # 奖励差异的统计
        reward_margin = implicit_rewards.mean().item()
        reward_std = implicit_rewards.std().item()
        
        # 策略和参考模型的 log 概率差异
        policy_margin = policy_log_ratio.mean().item()
        ref_margin = ref_log_ratio.mean().item()
    
    metrics = {
        'accuracy': accuracy,
        'reward_margin': reward_margin,
        'reward_std': reward_std,
        'policy_log_ratio_mean': policy_margin,
        'ref_log_ratio_mean': ref_margin,
        'chosen_log_probs_mean': policy_chosen_log_probs.mean().item(),
        'rejected_log_probs_mean': policy_rejected_log_probs.mean().item()
    }
    
    return loss, metrics


def get_batch_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    从 logits 和 labels 计算平均 log 概率
    
    用于计算整个序列的 log 概率（长度归一化）
    
    Args:
        logits: 模型输出 logits [batch_size, seq_len, vocab_size]
        labels: 标签 token IDs [batch_size, seq_len]
        label_smoothing: 标签平滑
    
    Returns:
        平均 log 概率 [batch_size]
    """
    # 计算 log softmax
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # 获取每个位置对应 label 的 log 概率
    token_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)  # [batch_size, seq_len]
    
    # 计算平均 log 概率（长度归一化）
    # 排除 padding (假设 padding 的 label 为 -100 或 0)
    mask = (labels != -100) & (labels != 0)
    mask = mask.float()
    
    # 长度归一化的平均 log 概率
    sum_log_probs = (token_log_probs * mask).sum(dim=1)
    lengths = mask.sum(dim=1)
    
    # 避免除零
    lengths = torch.clamp(lengths, min=1)
    avg_log_probs = sum_log_probs / lengths
    
    return avg_log_probs


class DPOTrainer:
    """
    DPO 训练器
    
    实现完整的 DPO 训练流程：
    - 前向传播（策略模型和参考模型）
    - DPO 损失计算
    - 反向传播和优化
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        初始化 DPO 训练器
        
        Args:
            policy_model: 策略模型（可训练）
            ref_model: 参考模型（固定）
            optimizer: 优化器
            config: 配置字典，包含：
                - beta: DPO 温度参数
                - label_smoothing: 标签平滑
                - loss_type: 损失类型 ('sigmoid' 或 'hinge')
            device: 训练设备
        """
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # 从配置中读取超参数
        self.beta = config.get('beta', 0.1)
        self.label_smoothing = config.get('label_smoothing', 0.0)
        self.loss_type = config.get('loss_type', 'sigmoid')
        
        # 冻结参考模型
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        logger.info(f"DPOTrainer 初始化完成，beta={self.beta}")
    
    @torch.no_grad()
    def compute_ref_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算参考模型的 log 概率（不计算梯度）
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
        
        Returns:
            平均 log 概率
        """
        outputs = self.ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        log_probs = get_batch_log_probs(logits, input_ids)
        
        return log_probs
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        单步 DPO 训练
        
        Args:
            batch: 数据批次，包含：
                - chosen_input_ids, chosen_attention_mask
                - rejected_input_ids, rejected_attention_mask
        
        Returns:
            loss: DPO 损失
            metrics: 指标字典
        """
        self.policy_model.train()
        
        # 将数据移动到设备
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # ===== 策略模型前向传播 =====
        # Chosen
        policy_chosen_outputs = self.policy_model(
            input_ids=batch['chosen_input_ids'],
            attention_mask=batch['chosen_attention_mask']
        )
        policy_chosen_log_probs = get_batch_log_probs(
            policy_chosen_outputs.logits,
            batch['chosen_input_ids']
        )
        
        # Rejected
        policy_rejected_outputs = self.policy_model(
            input_ids=batch['rejected_input_ids'],
            attention_mask=batch['rejected_attention_mask']
        )
        policy_rejected_log_probs = get_batch_log_probs(
            policy_rejected_outputs.logits,
            batch['rejected_input_ids']
        )
        
        # ===== 参考模型前向传播（无梯度）=====
        with torch.no_grad():
            ref_chosen_log_probs = self.compute_ref_log_probs(
                batch['chosen_input_ids'],
                batch['chosen_attention_mask']
            )
            ref_rejected_log_probs = self.compute_ref_log_probs(
                batch['rejected_input_ids'],
                batch['rejected_attention_mask']
            )
        
        # ===== 计算 DPO 损失 =====
        loss, metrics = compute_dpo_loss(
            policy_chosen_log_probs=policy_chosen_log_probs,
            policy_rejected_log_probs=policy_rejected_log_probs,
            ref_chosen_log_probs=ref_chosen_log_probs,
            ref_rejected_log_probs=ref_rejected_log_probs,
            beta=self.beta,
            label_smoothing=self.label_smoothing
        )
        
        # ===== 反向传播 =====
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        
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
        total_metrics = None
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            loss, metrics = self.train_step(batch)
            
            total_loss += loss.item()
            
            if total_metrics is None:
                total_metrics = {k: 0.0 for k in metrics.keys()}
            for k, v in metrics.items():
                total_metrics[k] += v
            
            num_batches += 1
            
            # 日志
            if (step + 1) % logging_steps == 0:
                avg_loss = total_loss / num_batches
                avg_accuracy = total_metrics['accuracy'] / num_batches
                logger.info(
                    f"Epoch {epoch}, Step {step + 1}, "
                    f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}"
                )
        
        # 计算平均指标
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        avg_metrics['train_loss'] = avg_loss
        
        return avg_metrics
    
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
        self.policy_model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            policy_chosen_outputs = self.policy_model(
                input_ids=batch['chosen_input_ids'],
                attention_mask=batch['chosen_attention_mask']
            )
            policy_chosen_log_probs = get_batch_log_probs(
                policy_chosen_outputs.logits,
                batch['chosen_input_ids']
            )
            
            policy_rejected_outputs = self.policy_model(
                input_ids=batch['rejected_input_ids'],
                attention_mask=batch['rejected_attention_mask']
            )
            policy_rejected_log_probs = get_batch_log_probs(
                policy_rejected_outputs.logits,
                batch['rejected_input_ids']
            )
            
            ref_chosen_log_probs = self.compute_ref_log_probs(
                batch['chosen_input_ids'],
                batch['chosen_attention_mask']
            )
            ref_rejected_log_probs = self.compute_ref_log_probs(
                batch['rejected_input_ids'],
                batch['rejected_attention_mask']
            )
            
            # 计算损失
            loss, metrics = compute_dpo_loss(
                policy_chosen_log_probs=policy_chosen_log_probs,
                policy_rejected_log_probs=policy_rejected_log_probs,
                ref_chosen_log_probs=ref_chosen_log_probs,
                ref_rejected_log_probs=ref_rejected_log_probs,
                beta=self.beta
            )
            
            total_loss += loss.item()
            total_accuracy += metrics['accuracy']
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
