"""
Policy Model 实现

策略模型是强化学习中的 Actor，负责生成动作（token）
基于预训练的语言模型
"""

from typing import Dict, Optional, Tuple, Any, List
import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import PreTrainedModel, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)


class PolicyModel(nn.Module):
    """
    策略模型
    
    基于预训练的语言模型，用于生成文本
    在 PPO 中作为 Actor，在 DPO/ORPO/SimPO 中作为策略模型
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        pad_token_id: int = 0
    ):
        """
        初始化策略模型
        
        Args:
            base_model: 预训练语言模型
            pad_token_id: Padding token ID
        """
        super().__init__()
        
        self.base_model = base_model
        self.pad_token_id = pad_token_id
        self.config = base_model.config
        
        logger.info(f"Policy Model 初始化完成")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码
            position_ids: 位置 IDs
            past_key_values: 缓存的 key/value states
            use_cache: 是否使用缓存
            output_hidden_states: 是否输出隐藏状态
            output_attentions: 是否输出注意力权重
            return_dict: 是否返回字典格式
        
        Returns:
            模型输出字典，包含 logits、hidden_states 等
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict
        )
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        生成文本
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: Top-p 采样参数
            top_k: Top-k 采样参数
            do_sample: 是否采样
            pad_token_id: Padding token ID
            eos_token_id: EOS token ID
        
        Returns:
            生成的 token IDs
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        # 使用 base_model 的 generate 方法
        generated = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        return generated
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        计算 log 概率
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            labels: 标签（用于计算损失）
        
        Returns:
            log_probs: 每个 token 的 log 概率
            loss: 如果提供了 labels，返回损失
        """
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # 计算 log softmax
        log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        
        loss = None
        if labels is not None:
            # 计算交叉熵损失
            # 将 labels 移位，因为预测的是下一个 token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # 应用 mask
            loss = loss.view(shift_labels.size())
            loss = (loss * shift_mask).sum() / shift_mask.sum()
        
        return log_probs, loss
    
    def get_per_token_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算每个 token 的 log 概率（用于 SimPO 等算法）
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
        
        Returns:
            每个样本的平均 log 概率 [batch_size]
        """
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # 获取每个位置的 log 概率（对应下一个 token）
        # 对于位置 i，我们看预测 token i+1 的概率
        selected_log_probs = torch.gather(
            log_probs[:, :-1, :],
            dim=2,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        # 应用 mask（排除 padding）
        mask = attention_mask[:, 1:]
        selected_log_probs = selected_log_probs * mask
        
        # 计算平均 log 概率（长度归一化）
        sum_log_probs = selected_log_probs.sum(dim=1)
        lengths = mask.sum(dim=1)
        
        # 避免除零
        lengths = torch.clamp(lengths, min=1)
        avg_log_probs = sum_log_probs / lengths
        
        return avg_log_probs


class ValueModel(nn.Module):
    """
    价值模型（Critic）
    
    用于 PPO 算法，估计状态价值 V(s)
    基于语言模型，输出标量价值
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        初始化价值模型
        
        Args:
            base_model: 预训练语言模型
            hidden_dim: 隐藏层维度
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.base_model = base_model
        model_hidden_size = getattr(base_model.config, 'hidden_size', 768)
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(model_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播，计算状态价值
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
        
        Returns:
            价值估计 [batch_size]
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 获取最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]
        
        # 使用最后一个 token 的隐藏状态
        last_hidden = hidden_states[:, -1, :]
        
        # 计算价值
        values = self.value_head(last_hidden).squeeze(-1)
        
        return values


def create_policy_model(
    model_name: str,
    pad_token_id: int = 0
) -> PolicyModel:
    """
    创建策略模型工厂函数
    
    Args:
        model_name: 预训练模型名称或路径
        pad_token_id: Padding token ID
    
    Returns:
        PolicyModel 实例
    """
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    policy_model = PolicyModel(
        base_model=base_model,
        pad_token_id=pad_token_id
    )
    
    return policy_model


def create_value_model(
    model_name: str,
    hidden_dim: int = 256,
    dropout: float = 0.1
) -> ValueModel:
    """
    创建价值模型工厂函数
    
    Args:
        model_name: 预训练模型名称或路径
        hidden_dim: 隐藏层维度
        dropout: Dropout 比例
    
    Returns:
        ValueModel 实例
    """
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    value_model = ValueModel(
        base_model=base_model,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    return value_model
