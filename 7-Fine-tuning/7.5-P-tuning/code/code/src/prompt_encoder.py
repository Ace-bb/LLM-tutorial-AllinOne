# -*- coding: utf-8 -*-
"""
P-tuning 提示编码器模块

实现 LSTM+MLP 架构的提示编码器，用于生成虚拟词元的连续向量表示。
这是 P-tuning 的核心组件，负责将可学习的虚拟词元映射到预训练模型的嵌入空间。
"""

import torch
import torch.nn as nn
from typing import Optional


class PromptEncoder(nn.Module):
    """
    P-tuning 提示编码器（LSTM + MLP 架构）
    
    架构说明：
    1. 虚拟词元嵌入层：可学习的连续向量（维度=模型嵌入维度）
    2. LSTM 编码器：捕捉虚拟词元之间的依赖关系（双向）
    3. MLP 映射层：将 LSTM 输出映射到模型嵌入空间
    
    输入：无（虚拟词元是模型参数的一部分）
    输出：[batch_size, num_virtual_tokens, embed_dim] 的提示嵌入
    
    参考论文：《GPT Understands, Too》(ACL 2021)
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        num_virtual_tokens: int = 50,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0
    ):
        """
        初始化提示编码器
        
        Args:
            embed_dim: 嵌入维度（必须与预训练模型一致）
            hidden_dim: LSTM 隐藏层维度（默认 512）
            num_virtual_tokens: 虚拟词元数量（默认 50）
            num_layers: LSTM 层数（默认 1）
            bidirectional: 是否使用双向 LSTM（默认 True）
            dropout: Dropout 比例（默认 0.0）
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_virtual_tokens = num_virtual_tokens
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # ========== 1. 虚拟词元嵌入层 ==========
        # 这些是可学习的参数，不是词汇表中的真实 token
        # 形状：[num_virtual_tokens, embed_dim]
        self.virtual_tokens = nn.Embedding(num_virtual_tokens, embed_dim)
        
        # 初始化：正态分布（与 BERT/GPT 初始化一致）
        nn.init.normal_(self.virtual_tokens.weight, mean=0.0, std=0.02)
        
        # ========== 2. LSTM 编码器 ==========
        # 捕捉虚拟词元之间的依赖关系
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # ========== 3. MLP 映射层 ==========
        # 将 LSTM 输出映射回模型嵌入空间
        # 输入维度：hidden_dim * num_directions（双向则为 2 倍）
        # 输出维度：embed_dim（与模型嵌入维度一致）
        mlp_input_dim = hidden_dim * self.num_directions
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        生成提示嵌入
        
        Args:
            batch_size: 当前批次的样本数量
        
        Returns:
            prompt_embeds: 提示嵌入 [batch_size, num_virtual_tokens, embed_dim]
        """
        # ========== 步骤 1: 获取虚拟词元嵌入 ==========
        # virtual_tokens.weight: [num_virtual_tokens, embed_dim]
        # unsqueeze(0): [1, num_virtual_tokens, embed_dim]
        # expand: [batch_size, num_virtual_tokens, embed_dim]
        token_embeds = self.virtual_tokens.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # ========== 步骤 2: LSTM 编码 ==========
        # 输入：[batch_size, num_virtual_tokens, embed_dim]
        # 输出：lstm_out [batch_size, num_virtual_tokens, hidden_dim*num_directions]
        lstm_out, _ = self.lstm(token_embeds)
        
        # ========== 步骤 3: MLP 映射 ==========
        # 将 LSTM 输出映射到模型嵌入空间
        # 输出：[batch_size, num_virtual_tokens, embed_dim]
        prompt_embeds = self.mlp(lstm_out)
        
        return prompt_embeds
    
    def get_virtual_tokens(self) -> torch.Tensor:
        """
        获取虚拟词元的原始嵌入（未经过编码器）
        
        Returns:
            [num_virtual_tokens, embed_dim] 的虚拟词元嵌入
        """
        return self.virtual_tokens.weight
    
    def save_prompt(self, path: str):
        """
        保存提示编码器参数（用于推理时加载）
        
        Args:
            path: 保存路径
        """
        torch.save({
            'virtual_tokens': self.virtual_tokens.weight.detach().cpu(),
            'lstm_state_dict': self.lstm.state_dict(),
            'mlp_state_dict': self.mlp.state_dict(),
            'config': {
                'embed_dim': self.embed_dim,
                'hidden_dim': self.hidden_dim,
                'num_virtual_tokens': self.num_virtual_tokens,
                'num_layers': self.num_layers,
                'bidirectional': self.bidirectional,
            }
        }, path)
    
    @classmethod
    def load_prompt(cls, path: str, device: str = 'cpu') -> 'PromptEncoder':
        """
        从文件加载提示编码器
        
        Args:
            path: 文件路径
            device: 加载设备
        
        Returns:
            加载好的 PromptEncoder 实例
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        encoder = cls(
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            num_virtual_tokens=config['num_virtual_tokens'],
            num_layers=config['num_layers'],
            bidirectional=config['bidirectional']
        )
        
        encoder.virtual_tokens.weight.data = checkpoint['virtual_tokens']
        encoder.lstm.load_state_dict(checkpoint['lstm_state_dict'])
        encoder.mlp.load_state_dict(checkpoint['mlp_state_dict'])
        
        return encoder.to(device)


class SimplePromptEncoder(nn.Module):
    """
    简化的提示编码器（无 LSTM，直接 MLP）
    
    用于对比实验或资源受限场景。
    参考：Prompt Tuning (Lester et al., 2021) 的方法
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_virtual_tokens: int = 50,
        hidden_dim: int = 512
    ):
        """
        初始化简化编码器
        
        Args:
            embed_dim: 嵌入维度
            num_virtual_tokens: 虚拟词元数量
            hidden_dim: 隐藏层维度（可选，用于中间映射）
        """
        super().__init__()
        
        self.num_virtual_tokens = num_virtual_tokens
        
        # 直接学习虚拟词元嵌入（无编码器）
        self.virtual_tokens = nn.Embedding(num_virtual_tokens, embed_dim)
        nn.init.normal_(self.virtual_tokens.weight, mean=0.0, std=0.02)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        生成提示嵌入
        
        Args:
            batch_size: 批次大小
        
        Returns:
            prompt_embeds: [batch_size, num_virtual_tokens, embed_dim]
        """
        # 直接返回虚拟词元嵌入（无编码）
        prompt_embeds = self.virtual_tokens.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        return prompt_embeds


if __name__ == "__main__":
    # ========== 测试代码 ==========
    print("测试 PromptEncoder...")
    
    # 创建编码器（GPT-2 配置）
    encoder = PromptEncoder(
        embed_dim=768,  # GPT-2 嵌入维度
        hidden_dim=512,
        num_virtual_tokens=50,
        bidirectional=True
    )
    
    print(f"编码器参数:")
    print(f"  嵌入维度：{encoder.embed_dim}")
    print(f"  隐藏层维度：{encoder.hidden_dim}")
    print(f"  虚拟词元数量：{encoder.num_virtual_tokens}")
    print(f"  双向 LSTM: {encoder.bidirectional}")
    
    # 测试前向传播
    batch_size = 4
    prompt_embeds = encoder(batch_size)
    
    print(f"\n前向传播测试:")
    print(f"  输入批次大小：{batch_size}")
    print(f"  输出形状：{prompt_embeds.shape}")
    print(f"  期望形状：[4, 50, 768]")
    
    # 验证形状
    assert prompt_embeds.shape == (batch_size, encoder.num_virtual_tokens, encoder.embed_dim)
    print("\n✓ 测试通过！")
    
    # 打印参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\n参数量统计:")
    print(f"  总参数：{total_params:,}")
    print(f"  可训练参数：{trainable_params:,}")
