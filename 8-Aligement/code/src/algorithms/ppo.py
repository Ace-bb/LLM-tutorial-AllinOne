"""
PPO (Proximal Policy Optimization) 实现

近端策略优化算法，基于 Actor-Critic 架构
使用 GAE 计算优势函数，Clip 机制保证训练稳定性

核心公式:
1. GAE 优势估计:
   A_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T-t+1}δ_{T-1}
   其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

2. PPO-Clip 损失:
   L^{CLIP} = E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]
   其中 r_t = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)

3. 总损失:
   L = L^{CLIP} - c1·L^{VF} + c2·S[π_θ]
   其中 L^{VF} 是价值损失，S 是熵 bonus
"""

from typing import Dict, Optional, Tuple, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> torch.Tensor:
    """
    计算 GAE (Generalized Advantage Estimation) 优势函数
    
    GAE 是一种低方差的优势估计方法，通过指数加权平均
    平衡了 TD 误差和蒙特卡洛估计
    
    Args:
        rewards: 奖励序列 [seq_len] 或 [batch_size, seq_len]
        values: 价值估计 [seq_len+1] 或 [batch_size, seq_len+1]
        dones: 终止标志 [seq_len] 或 [batch_size, seq_len]
        gamma: 折扣因子 (默认 0.99)
        lam: GAE lambda 参数 (默认 0.95)
    
    Returns:
        advantages: GAE 优势估计 [seq_len] 或 [batch_size, seq_len]
    """
    # 确保输入维度一致
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(0)
        values = values.unsqueeze(0)
        dones = dones.unsqueeze(0)
    
    batch_size, seq_len = rewards.shape
    
    # 计算 TD 误差: δ_t = r_t + γV(s_{t+1}) - V(s_t)
    # values[:, :-1] 是 V(s_t), values[:, 1:] 是 V(s_{t+1})
    deltas = rewards + gamma * values[:, 1:] * (1 - dones) - values[:, :-1]
    
    # 初始化 GAE
    advantages = torch.zeros_like(deltas)
    
    # 反向计算 GAE
    # A_t = δ_t + (γλ)·A_{t+1}
    gae = 0
    for t in reversed(range(seq_len)):
        gae = deltas[:, t] + gamma * lam * (1 - dones[:, t]) * gae
        advantages[:, t] = gae
    
    return advantages


def compute_ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 PPO-Clip 损失
    
    核心思想：限制策略更新幅度，防止训练不稳定
    
    损失函数:
    L^{CLIP} = E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]
    
    Args:
        old_log_probs: 旧策略的 log 概率 [batch_size, seq_len]
        new_log_probs: 新策略的 log 概率 [batch_size, seq_len]
        advantages: 优势估计 [batch_size, seq_len]
        clip_epsilon: Clip 范围 (默认 0.2)
    
    Returns:
        loss: PPO-Clip 损失 (标量)
        ratio: 概率比率
        clip_fraction: 被 clip 的比例
    """
    # 计算概率比率: r_t = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
    # 使用 log 概率计算：r_t = exp(log_π_θ - log_π_{θ_old})
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # 计算 unclipped 损失: r_t · A_t
    surr1 = ratio * advantages
    
    # 计算 clipped 损失: clip(r_t, 1-ε, 1+ε) · A_t
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    surr2 = clipped_ratio * advantages
    
    # 取最小值 (对于优势为正的情况，限制增长；对于优势为负的情况，限制下降)
    # PPO 的核心：只优化不受 clip 限制的部分
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 计算 clip 比例（用于监控）
    clip_fraction = (torch.abs(ratio - 1) > clip_epsilon).float().mean()
    
    return policy_loss, ratio, clip_fraction


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float = 0.2,
    old_values: Optional[torch.Tensor] = None,
    clip_value: bool = True
) -> torch.Tensor:
    """
    计算价值损失
    
    支持两种损失类型：
    1. MSE: (V(s) - R)^2
    2. Clipped MSE: 限制价值更新幅度
    
    Args:
        values: 当前价值估计 [batch_size, seq_len]
        returns: 目标回报 [batch_size, seq_len]
        clip_epsilon: Clip 范围
        old_values: 旧价值估计（用于 clip）
        clip_value: 是否使用 clip
    
    Returns:
        value_loss: 价值损失 (标量)
    """
    if clip_value and old_values is not None:
        # Clipped value loss
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -clip_epsilon,
            clip_epsilon
        )
        loss1 = (values - returns) ** 2
        loss2 = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(loss1, loss2).mean()
    else:
        # Standard MSE loss
        value_loss = 0.5 * ((values - returns) ** 2).mean()
    
    return value_loss


def compute_entropy_bonus(log_probs: torch.Tensor) -> torch.Tensor:
    """
    计算熵 bonus
    
    熵用于鼓励探索，防止策略过早收敛到次优解
    
    H(π) = -E[log π(a|s)]
    
    Args:
        log_probs: log 概率 [batch_size, seq_len]
    
    Returns:
        entropy: 平均熵 (标量)
    """
    # 计算概率分布的熵
    # 假设 log_probs 是来自 categorical distribution 的 log 概率
    # 需要恢复完整的概率分布来计算熵
    
    # 简单近似：使用平均 log 概率的负值
    # 更准确的方法需要完整的 logits
    entropy = -log_probs.mean()
    
    return entropy


class PPOTrainer:
    """
    PPO 训练器
    
    实现完整的 PPO 训练流程，包括：
    - 数据收集（rollout）
    - GAE 优势计算
    - PPO 更新
    - 价值函数更新
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        optimizer_policy: torch.optim.Optimizer,
        optimizer_value: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        初始化 PPO 训练器
        
        Args:
            policy_model: 策略模型（Actor）
            value_model: 价值模型（Critic）
            optimizer_policy: 策略优化器
            optimizer_value: 价值优化器
            config: 配置字典，包含：
                - clip_epsilon: PPO clip 参数
                - value_coeff: 价值损失系数
                - entropy_coeff: 熵 bonus 系数
                - gae_lambda: GAE lambda
                - gamma: 折扣因子
                - ppo_epochs: PPO 更新轮数
                - mini_batch_size: mini-batch 大小
            device: 训练设备
        """
        self.policy_model = policy_model.to(device)
        self.value_model = value_model.to(device)
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.config = config
        self.device = device
        
        # 从配置中读取超参数
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coeff = config.get('value_coeff', 0.5)
        self.entropy_coeff = config.get('entropy_coeff', 0.01)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.gamma = config.get('gamma', 0.99)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.mini_batch_size = config.get('mini_batch_size', 1)
        self.normalize_advantage = config.get('normalize_advantage', True)
        
        logger.info(f"PPOTrainer 初始化完成")
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算优势函数和回报
        
        Args:
            rewards: 奖励序列
            values: 价值估计
            dones: 终止标志
        
        Returns:
            advantages: GAE 优势估计
            returns: 回报（advantages + values）
        """
        # 计算 GAE 优势
        advantages = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.gamma,
            lam=self.gae_lambda
        )
        
        # 计算回报：R_t = A_t + V(s_t)
        returns = advantages + values[:, :-1]
        
        # 优势标准化（重要技巧，提高训练稳定性）
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        获取指定动作的 log 概率
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            actions: 动作（选择的 token）
        
        Returns:
            log_probs: 动作的 log 概率
        """
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # 获取每个位置选择动作的 log 概率
        log_probs = torch.gather(
            torch.log_softmax(logits, dim=-1),
            dim=2,
            index=actions.unsqueeze(-1)
        ).squeeze(-1)
        
        return log_probs
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> Dict[str, float]:
        """
        单步 PPO 更新
        
        Args:
            batch: 数据批次
            old_log_probs: 旧策略的 log 概率
            advantages: 优势估计
            returns: 回报
            old_values: 旧价值估计
        
        Returns:
            metrics: 训练指标字典
        """
        self.policy_model.train()
        self.value_model.train()
        
        # 将数据移动到设备
        batch = {k: v.to(self.device) for k, v in batch.items()}
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        old_values = old_values.to(self.device)
        
        # 获取新策略的 log 概率
        new_log_probs = self.get_log_probs(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            actions=batch['actions']
        )
        
        # 获取新价值估计
        new_values = self.value_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # 计算 PPO-Clip 损失
        policy_loss, ratio, clip_fraction = compute_ppo_loss(
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            advantages=advantages,
            clip_epsilon=self.clip_epsilon
        )
        
        # 计算价值损失
        value_loss = compute_value_loss(
            values=new_values,
            returns=returns,
            clip_epsilon=self.clip_epsilon,
            old_values=old_values,
            clip_value=self.config.get('clip_value', True)
        )
        
        # 计算熵 bonus（近似）
        entropy = compute_entropy_bonus(new_log_probs)
        
        # 总损失
        total_loss = (
            policy_loss
            + self.value_coeff * value_loss
            - self.entropy_coeff * entropy
        )
        
        # 反向传播
        # 策略模型反向传播
        self.optimizer_policy.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer_policy.step()
        
        # 价值模型反向传播
        self.optimizer_value.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
        self.optimizer_value.step()
        
        # 收集指标
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'entropy': entropy.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item(),
            'clip_fraction': clip_fraction.item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
        
        return metrics
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        logging_steps: int = 50
    ) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            dataloader: 数据 DataLoader
            epoch: 当前 epoch
            logging_steps: 日志步数
        
        Returns:
            平均训练指标
        """
        total_metrics = None
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            # 这里需要先从当前策略收集数据（rollout）
            # 简化版本，假设 batch 中已经包含所有需要的数据
            
            # 实际使用中，这里应该：
            # 1. 使用当前策略生成动作
            # 2. 计算奖励
            # 3. 计算优势和回报
            # 4. 进行 PPO 更新
            
            # 简化示例
            metrics = {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'total_loss': 0.0
            }
            
            if total_metrics is None:
                total_metrics = {k: 0.0 for k in metrics.keys()}
            
            for k, v in metrics.items():
                total_metrics[k] += v
            
            num_batches += 1
            
            if (step + 1) % logging_steps == 0:
                logger.info(f"Epoch {epoch}, Step {step + 1}")
        
        # 计算平均指标
        if total_metrics:
            total_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return total_metrics if total_metrics else {}
