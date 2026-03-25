"""
Reward Model 实现

基于 Transformer 的分类头，实现 Bradley-Terry 模型
用于预测给定 prompt 下，哪个回复更好
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    奖励模型
    
    基于预训练 Transformer 模型，添加分类头
    使用 Bradley-Terry 模型计算偏好概率
    
    Bradley-Terry 模型:
    P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    
    其中 r_chosen 和 r_rejected 分别是 chosen 和 rejected 的奖励分数
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """
        初始化奖励模型
        
        Args:
            base_model: 预训练 Transformer 模型
            hidden_dim: 分类头隐藏层维度
            dropout: Dropout 比例
            pad_token_id: Padding token ID
        """
        super().__init__()
        
        self.base_model = base_model
        self.pad_token_id = pad_token_id
        
        # 获取隐藏层维度
        model_hidden_size = getattr(base_model.config, 'hidden_size', 768)
        
        # 分类头：将 [EOS] token 的隐藏状态映射到标量奖励
        self.reward_head = nn.Sequential(
            nn.Linear(model_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        logger.info(f"Reward Model 初始化完成，隐藏层维度：{hidden_dim}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            return_dict: 是否返回字典格式
        
        Returns:
            奖励分数 [batch_size] 或字典
        """
        # 获取 base model 输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 获取最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # 获取每个序列的最后一个非 padding token 的隐藏状态
        # 方法：使用 attention_mask 找到每个样本的有效长度
        if attention_mask is not None:
            # 计算每个样本的有效长度
            lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
            # 获取最后一个有效 token 的索引
            last_indices = lengths.unsqueeze(1).unsqueeze(2).expand(-1, -1, hidden_states.size(-1))
            # 提取对应的隐藏状态
            last_hidden = torch.gather(hidden_states, 1, last_indices).squeeze(1)  # [batch_size, hidden_size]
        else:
            # 如果没有 attention_mask，使用最后一个 token
            last_hidden = hidden_states[:, -1, :]
        
        # 通过分类头得到奖励分数
        rewards = self.reward_head(last_hidden).squeeze(-1)  # [batch_size]
        
        if return_dict:
            return {'rewards': rewards}
        return rewards
    
    def compute_pairwise_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 Bradley-Terry 成对损失
        
        损失函数：
        L = -log(sigmoid(r_chosen - r_rejected))
        
        Args:
            chosen_input_ids: chosen 回复的 input_ids
            chosen_attention_mask: chosen 的 attention mask
            rejected_input_ids: rejected 回复的 input_ids
            rejected_attention_mask: rejected 的 attention mask
        
        Returns:
            loss: 标量损失
            metrics: 指标字典
        """
        # 计算 chosen 和 rejected 的奖励
        chosen_outputs = self(chosen_input_ids, chosen_attention_mask)
        rejected_outputs = self(rejected_input_ids, rejected_attention_mask)
        
        chosen_rewards = chosen_outputs['rewards']
        rejected_rewards = rejected_outputs['rewards']
        
        # Bradley-Terry 损失
        # P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
        # loss = -log(P(chosen > rejected))
        logits = chosen_rewards - rejected_rewards
        loss = -nn.functional.logsigmoid(logits).mean()
        
        # 计算准确率
        with torch.no_grad():
            predictions = (chosen_rewards > rejected_rewards).float()
            accuracy = predictions.mean().item()
            
            # 计算奖励差异的均值
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        
        metrics = {
            'accuracy': accuracy,
            'reward_margin': reward_margin,
            'chosen_reward_mean': chosen_rewards.mean().item(),
            'rejected_reward_mean': rejected_rewards.mean().item()
        }
        
        return loss, metrics


class RewardModelTrainer:
    """
    奖励模型训练器
    
    实现完整的训练循环，包括：
    - 前向传播
    - 损失计算
    - 反向传播
    - 评估
    """
    
    def __init__(
        self,
        model: RewardModel,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        """
        初始化训练器
        
        Args:
            model: RewardModel 实例
            optimizer: 优化器
            device: 训练设备
            gradient_accumulation_steps: 梯度累积步数
            max_grad_norm: 最大梯度范数
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        单步训练
        
        Args:
            batch: 数据批次
        
        Returns:
            loss: 损失值
            metrics: 指标字典
        """
        self.model.train()
        
        # 将数据移动到设备
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 计算损失
        loss, metrics = self.model.compute_pairwise_loss(
            chosen_input_ids=batch['chosen_input_ids'],
            chosen_attention_mask=batch['chosen_attention_mask'],
            rejected_input_ids=batch['rejected_input_ids'],
            rejected_attention_mask=batch['rejected_attention_mask']
        )
        
        # 梯度累积归一化
        loss = loss / self.gradient_accumulation_steps
        
        # 反向传播
        loss.backward()
        
        return loss * self.gradient_accumulation_steps, metrics
    
    def optimizer_step(self) -> None:
        """
        执行优化器更新
        
        包括梯度裁剪和优化器步进
        """
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )
        
        # 更新参数
        self.optimizer.step()
        self.optimizer.zero_grad()
    
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
            
            loss, metrics = self.model.compute_pairwise_loss(
                chosen_input_ids=batch['chosen_input_ids'],
                chosen_attention_mask=batch['chosen_attention_mask'],
                rejected_input_ids=batch['rejected_input_ids'],
                rejected_attention_mask=batch['rejected_attention_mask']
            )
            
            total_loss += loss.item()
            total_accuracy += metrics['accuracy']
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
    
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
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            loss, metrics = self.train_step(batch)
            
            # 梯度累积后更新
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer_step()
            
            total_loss += loss.item()
            total_accuracy += metrics['accuracy']
            num_batches += 1
            
            # 日志
            if (step + 1) % logging_steps == 0:
                avg_loss = total_loss / num_batches
                avg_accuracy = total_accuracy / num_batches
                logger.info(
                    f"Epoch {epoch}, Step {step + 1}, "
                    f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}"
                )
        
        return {
            'train_loss': total_loss / num_batches,
            'train_accuracy': total_accuracy / num_batches
        }


def create_reward_model(
    base_model_name: str,
    hidden_dim: int = 256,
    dropout: float = 0.1
) -> RewardModel:
    """
    创建奖励模型工厂函数
    
    Args:
        base_model_name: 预训练模型名称或路径
        hidden_dim: 隐藏层维度
        dropout: Dropout 比例
    
    Returns:
        RewardModel 实例
    """
    # 加载预训练模型（不使用预训练的分类头）
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1
    )
    
    # 移除预训练的分类头，使用我们自定义的
    reward_model = RewardModel(
        base_model=base_model,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    return reward_model
