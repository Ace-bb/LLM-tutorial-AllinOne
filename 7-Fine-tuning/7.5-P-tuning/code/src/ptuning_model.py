# -*- coding: utf-8 -*-
"""
P-tuning 模型封装模块

将 PromptEncoder 与预训练语言模型（GPT-2/BERT）结合，
实现完整的 P-tuning 模型。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel
)

from .prompt_encoder import PromptEncoder


class PTuningModel(nn.Module):
    """
    P-tuning 模型封装类
    
    核心功能：
    1. 冻结预训练语言模型主体参数
    2. 集成 PromptEncoder 生成提示嵌入
    3. 将提示嵌入与输入嵌入拼接
    4. 支持分类和生成任务
    
    架构流程：
    输入 → PromptEncoder → 提示嵌入 ─┐
                                    ├→ 拼接 → 冻结的 LM → 输出
    实际输入 → 嵌入层 → 输入嵌入 ────┘
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        num_virtual_tokens: int = 50,
        encoder_hidden_dim: int = 512,
        num_labels: int = 2,
        task_type: str = "classification",
        bidirectional_encoder: bool = True
    ):
        """
        初始化 P-tuning 模型
        
        Args:
            model_name: 预训练模型名称（如 'gpt2', 'bert-base-uncased'）
            num_virtual_tokens: 虚拟词元数量（默认 50）
            encoder_hidden_dim: 编码器隐藏层维度（默认 512）
            num_labels: 分类标签数（仅分类任务需要）
            task_type: 任务类型 ('classification' 或 'causal_lm')
            bidirectional_encoder: 是否使用双向 LSTM 编码器
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_virtual_tokens = num_virtual_tokens
        self.task_type = task_type
        self.num_labels = num_labels
        
        # ========== 加载预训练模型 ==========
        if task_type == "classification":
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        elif task_type == "causal_lm":
            self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            raise ValueError(f"不支持的任务类型：{task_type}")
        
        # 获取模型配置
        self.config = self.base_model.config
        self.embed_dim = self.config.hidden_size
        
        # ========== 冻结主模型参数 ==========
        # P-tuning 的核心：仅训练提示编码器，冻结其他所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # ========== 创建提示编码器 ==========
        self.prompt_encoder = PromptEncoder(
            embed_dim=self.embed_dim,
            hidden_dim=encoder_hidden_dim,
            num_virtual_tokens=num_virtual_tokens,
            bidirectional=bidirectional_encoder
        )
        
        # ========== 分类头（仅分类任务） ==========
        if task_type == "classification":
            # 使用模型自带的分类头
            # 注意：分类头也需要训练（因为它与 num_labels 相关）
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签（用于计算损失）
            **kwargs: 其他传递给 base_model 的参数
        
        Returns:
            outputs: 包含 loss, logits 等的字典
        """
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device
        
        # ========== 步骤 1: 生成提示嵌入 ==========
        # [batch_size, num_virtual_tokens, embed_dim]
        prompt_embeds = self.prompt_encoder(batch_size)
        
        # ========== 步骤 2: 获取输入嵌入 ==========
        # 从 base_model 获取嵌入层
        if hasattr(self.base_model, 'get_input_embeddings'):
            input_embeds = self.base_model.get_input_embeddings()(input_ids)
        else:
            # BERT 等模型使用 embeddings.word_embeddings
            input_embeds = self.base_model.embeddings.word_embeddings(input_ids)
        
        # ========== 步骤 3: 拼接提示和输入 ==========
        # [batch_size, num_virtual_tokens + seq_len, embed_dim]
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        # ========== 步骤 4: 调整注意力掩码 ==========
        # 为虚拟词元创建全 1 的掩码
        prompt_attention_mask = torch.ones(
            (batch_size, self.num_virtual_tokens),
            device=device,
            dtype=attention_mask.dtype
        )
        # 拼接掩码
        combined_attention_mask = torch.cat(
            [prompt_attention_mask, attention_mask],
            dim=1
        )
        
        # ========== 步骤 5: 通过 base_model ==========
        # 使用 inputs_embeds 而非 input_ids（因为我们有拼接后的嵌入）
        outputs = self.base_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=labels,
            **kwargs
        )
        
        # ========== 步骤 6: 处理输出 ==========
        # 对于分类任务，logits 已经正确处理
        # 对于生成任务，需要跳过虚拟词元部分
        if self.task_type == "causal_lm":
            # 跳过虚拟词元部分的 logits
            # outputs.logits: [batch, num_virtual_tokens+seq_len, vocab_size]
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                outputs.logits = outputs.logits[:, self.num_virtual_tokens:, :]
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """
        文本生成（仅用于 causal_lm 任务）
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成 token 数
            num_return_sequences: 返回序列数量
            temperature: 采样温度
            top_p: nucleus sampling 参数
            **kwargs: 其他生成参数
        
        Returns:
            generated_ids: 生成的 token IDs
        """
        if self.task_type != "causal_lm":
            raise ValueError("generate() 仅适用于 causal_lm 任务")
        
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 生成提示嵌入
        prompt_embeds = self.prompt_encoder(batch_size)
        input_embeds = self.base_model.get_input_embeddings()(input_ids)
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        # 调整注意力掩码
        prompt_attention_mask = torch.ones(
            (batch_size, self.num_virtual_tokens),
            device=device,
            dtype=attention_mask.dtype
        )
        combined_attention_mask = torch.cat(
            [prompt_attention_mask, attention_mask],
            dim=1
        )
        
        # 使用 base_model 的 generate 方法
        # 注意：需要传递 inputs_embeds 而非 input_ids
        generated_ids = self.base_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        return generated_ids
    
    def get_trainable_parameters(self):
        """
        获取可训练参数（仅提示编码器和分类头）
        
        Returns:
            可训练参数的迭代器
        """
        params = []
        
        # 提示编码器参数（全部可训练）
        params.extend(self.prompt_encoder.parameters())
        
        # 分类任务：分类头也可训练
        if self.task_type == "classification":
            if hasattr(self.base_model, 'classifier'):
                params.extend(self.base_model.classifier.parameters())
        
        return iter(params)
    
    def count_trainable_parameters(self) -> int:
        """
        统计可训练参数数量
        
        Returns:
            可训练参数总数
        """
        return sum(p.numel() for p in self.get_trainable_parameters())
    
    def count_total_parameters(self) -> int:
        """
        统计总参数数量
        
        Returns:
            总参数数
        """
        return sum(p.numel() for p in self.parameters())
    
    def print_trainable_parameters(self):
        """打印可训练参数统计信息"""
        trainable = self.count_trainable_parameters()
        total = self.count_total_parameters()
        percentage = (trainable / total) * 100 if total > 0 else 0
        
        print(f"可训练参数：{trainable:,} ({percentage:.4f}%)")
        print(f"总参数：{total:,}")
        print(f"冻结参数：{total - trainable:,}")
    
    def save_pretrained(self, save_directory: str):
        """
        保存 P-tuning 模型（仅保存提示编码器，base_model 不需要保存）
        
        Args:
            save_directory: 保存目录
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存提示编码器
        self.prompt_encoder.save_prompt(os.path.join(save_directory, "prompt_encoder.pt"))
        
        # 保存配置
        config = {
            'model_name': self.model_name,
            'num_virtual_tokens': self.num_virtual_tokens,
            'encoder_hidden_dim': self.prompt_encoder.hidden_dim,
            'num_labels': self.num_labels,
            'task_type': self.task_type,
            'bidirectional_encoder': self.prompt_encoder.bidirectional,
        }
        
        with open(os.path.join(save_directory, "ptuning_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_pretrained(cls, save_directory: str, device: str = 'cpu') -> 'PTuningModel':
        """
        从目录加载 P-tuning 模型
        
        Args:
            save_directory: 模型目录
            device: 加载设备
        
        Returns:
            加载好的 PTuningModel 实例
        """
        import os
        import json
        
        # 加载配置
        with open(os.path.join(save_directory, "ptuning_config.json"), 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 创建模型
        model = cls(
            model_name=config['model_name'],
            num_virtual_tokens=config['num_virtual_tokens'],
            encoder_hidden_dim=config['encoder_hidden_dim'],
            num_labels=config['num_labels'],
            task_type=config['task_type'],
            bidirectional_encoder=config['bidirectional_encoder']
        )
        
        # 加载提示编码器
        model.prompt_encoder = PromptEncoder.load_prompt(
            os.path.join(save_directory, "prompt_encoder.pt"),
            device=device
        )
        
        return model.to(device)


if __name__ == "__main__":
    # ========== 测试代码 ==========
    print("测试 PTuningModel...")
    
    # 创建模型（使用小型 GPT-2）
    model = PTuningModel(
        model_name="gpt2",
        num_virtual_tokens=50,
        encoder_hidden_dim=512,
        task_type="causal_lm"
    )
    
    print(f"\n模型配置:")
    print(f"  基础模型：{model.model_name}")
    print(f"  虚拟词元数量：{model.num_virtual_tokens}")
    print(f"  任务类型：{model.task_type}")
    
    # 打印参数统计
    print(f"\n参数统计:")
    model.print_trainable_parameters()
    
    # 测试前向传播
    print(f"\n测试前向传播...")
    batch_size = 2
    seq_len = 32
    
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"  输入形状：{input_ids.shape}")
    print(f"  输出 logits 形状：{outputs.logits.shape}")
    print(f"  期望形状：[{batch_size}, {seq_len}, 50257]")
    
    print("\n✓ 测试通过！")
