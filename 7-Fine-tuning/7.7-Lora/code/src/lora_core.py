"""
LoRA 核心实现模块
================

本模块从零实现 LoRA (Low-Rank Adaptation) 的核心组件，
帮助理解 LoRA 的工作原理。

LoRA 的核心思想：
1. 冻结预训练模型的所有参数
2. 在原始权重矩阵 W 上添加低秩分解的增量矩阵 ΔW = BA
3. 只训练 A 和 B 矩阵，大幅减少可训练参数量

数学表达：
    W_updated = W + ΔW = W + BA
    
其中：
    - W ∈ R^(d×k) 是原始预训练权重（冻结）
    - B ∈ R^(d×r) 是 LoRA 的 B 矩阵（可训练）
    - A ∈ R^(r×k) 是 LoRA 的 A 矩阵（可训练）
    - r << min(d, k) 是秩（rank），控制参数量

作者：LoRA 技术文章示例代码
日期：2026-03-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


class LoRALayer(nn.Module):
    """
    LoRA 基础层 - 实现低秩适应的核心组件
    
    原理说明：
    --------
    传统全连接层：y = xW + b, 其中 W ∈ R^(d×k)
    
    LoRA 改造后：y = x(W + BA) + b
    - A 矩阵初始化为高斯分布 N(0, 1)
    - B 矩阵初始化为零矩阵
    - 这样保证训练开始时 ΔW = BA = 0，不改变原模型行为
    
    缩放因子：
    --------
    实际计算时会乘以缩放因子 alpha/r：
        y = xW + x(BA) * (alpha/r) + b
    
    这样设计的好处：
    - 调整 alpha 可以控制 LoRA 的影响程度
    - 保持学习率不变的情况下调整适配强度
    """
    
    def __init__(
        self, 
        in_features: int,      # 输入特征维度 (k)
        out_features: int,     # 输出特征维度 (d)
        r: int = 8,            # LoRA 秩，控制低秩矩阵的大小
        alpha: float = 16.0,   # 缩放因子
        dropout: float = 0.05, # Dropout 比例，防止过拟合
        bias: bool = False     # 是否使用偏置
    ):
        super().__init__()
        
        # 保存参数
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        # 计算缩放因子：alpha / r
        # 这是 LoRA 的关键设计，保证不同 r 值下的学习强度一致
        self.scaling = alpha / r
        
        # 定义 Dropout 层
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # ============ LoRA 的核心：A 和 B 矩阵 ============
        
        # A 矩阵：R^(r×k)，初始化为高斯分布
        # 形状：[r, in_features]
        # 使用高斯初始化，均值为 0，标准差为 1
        self.lora_A = nn.Parameter(
            torch.zeros((r, in_features)).normal_(mean=0.0, std=1.0)
        )
        
        # B 矩阵：R^(d×r)，初始化为零矩阵
        # 形状：[out_features, r]
        # 关键：初始化为 0，保证训练开始时 LoRA 不产生任何影响
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        
        # 可选的偏置项（通常不使用）
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 标记哪些参数需要训练
        # LoRA 只训练 A 和 B，原始权重 W 保持冻结
        self._mark_parameters()
    
    def _mark_parameters(self):
        """标记参数是否需要梯度"""
        # A 和 B 矩阵需要训练
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
        
        # 偏置（如果有）也需要训练
        if self.bias is not None:
            self.bias.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        计算过程：
        1. 输入 x 通过 A 矩阵：x @ A^T，形状从 [..., k] → [..., r]
        2. 通过 Dropout（训练时）
        3. 通过 B 矩阵：(x @ A^T) @ B^T，形状从 [..., r] → [..., d]
        4. 乘以缩放因子：result * (alpha/r)
        5. 加上偏置（如果有）
        
        参数:
            x: 输入张量，形状为 [..., in_features]
        
        返回:
            输出张量，形状为 [..., out_features]
        """
        # LoRA 计算：x @ A^T @ B^T * scaling
        # 使用矩阵乘法的结合律优化计算顺序
        # 先计算 x @ A^T 将维度从 k 降到 r，再 @ B^T 升到 d
        
        result = self.dropout(x) @ self.lora_A.transpose(0, 1)  # [..., r]
        result = result @ self.lora_B.transpose(0, 1)           # [..., d]
        result = result * self.scaling                           # 缩放
        
        # 加上偏置
        if self.bias is not None:
            result = result + self.bias
        
        return result
    
    def merge_weights(self) -> torch.Tensor:
        """
        合并 LoRA 权重到原始权重
        
        训练完成后，可以将 BA 合并到原始权重 W 中：
            W_merged = W + BA * (alpha/r)
        
        这样做的好处：
        - 推理时不需要额外的矩阵乘法
        - 不增加任何推理延迟
        - 只需保存小的 LoRA 权重（MB 级别）
        
        返回:
            合并后的增量权重 ΔW = BA * scaling
        """
        # 计算 BA 矩阵乘积
        # B: [out_features, r], A: [r, in_features]
        # 结果：[out_features, in_features]，与原始 W 形状相同
        delta_weight = self.lora_B @ self.lora_A
        
        # 应用缩放因子
        delta_weight = delta_weight * self.scaling
        
        return delta_weight
    
    def get_param_count(self) -> int:
        """计算 LoRA 层的可训练参数量"""
        # A 矩阵：r × in_features
        # B 矩阵：out_features × r
        # 偏置：out_features（如果有）
        params = self.r * self.in_features + self.out_features * self.r
        if self.bias is not None:
            params += self.out_features
        return params


class LoRALinear(nn.Module):
    """
    带有 LoRA 的完整线性层
    
    这个类将原始线性层和 LoRA 层结合起来，
    模拟 PEFT 库中 LoRA 的工作方式。
    
    结构：
        output = x @ W^T + x @ (BA)^T * scaling + bias
               = x @ (W + BA * scaling)^T + bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        bias: bool = True,
        freeze_base: bool = True  # 是否冻结基础权重
    ):
        super().__init__()
        
        # 原始线性层（预训练权重）
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA 层（低秩适配）
        self.lora_layer = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=False  # LoRA 层通常不需要偏置
        )
        
        # 冻结基础权重（关键！）
        if freeze_base:
            self._freeze_base_weights()
        
        # 保存配置信息
        self.r = r
        self.alpha = alpha
    
    def _freeze_base_weights(self):
        """冻结基础层的所有参数"""
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        计算：base_output + lora_output
        """
        # 基础层输出（冻结的预训练权重）
        base_output = self.base_layer(x)
        
        # LoRA 适配输出（可训练的低秩矩阵）
        lora_output = self.lora_layer(x)
        
        # 相加得到最终输出
        return base_output + lora_output
    
    def get_trainable_param_count(self) -> int:
        """获取可训练参数量（仅 LoRA 部分）"""
        return self.lora_layer.get_param_count()
    
    def get_total_param_count(self) -> int:
        """获取总参数量"""
        total = sum(p.numel() for p in self.parameters())
        return total


class LoRAEmbedding(nn.Module):
    """
    带有 LoRA 的嵌入层
    
    用于对嵌入矩阵进行低秩适配，
    常用于 NLP 任务的输入层。
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 8,
        alpha: float = 16.0,
        freeze_base: bool = True
    ):
        super().__init__()
        
        # 基础嵌入层
        self.base_embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # LoRA 矩阵
        # A: [r, embedding_dim]
        # B: [num_embeddings, r]
        self.lora_A = nn.Parameter(
            torch.zeros((r, embedding_dim)).normal_(mean=0.0, std=1.0)
        )
        self.lora_B = nn.Parameter(torch.zeros((num_embeddings, r)))
        
        self.scaling = alpha / r
        self.r = r
        
        # 冻结基础嵌入
        if freeze_base:
            self.base_embedding.weight.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        对于输入 token IDs，计算：
        embedding = base_embed + lora_embed * scaling
        """
        # 基础嵌入
        base_embed = self.base_embedding(input_ids)  # [batch, seq_len, dim]
        
        # LoRA 嵌入
        # 查找对应的 B 矩阵行，然后乘以 A
        lora_B_selected = self.lora_B[input_ids]  # [batch, seq_len, r]
        lora_embed = lora_B_selected @ self.lora_A  # [batch, seq_len, dim]
        lora_embed = lora_embed * self.scaling
        
        return base_embed + lora_embed


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    将 LoRA 应用到模型的指定模块
    
    这是模拟 PEFT 库的 get_peft_model 函数的简化版本。
    
    参数:
        model: 原始 PyTorch 模型
        r: LoRA 秩
        alpha: 缩放因子
        dropout: Dropout 比例
        target_modules: 要应用 LoRA 的模块名列表
                       如 ["q_proj", "v_proj"] 或 ["linear"]
    
    返回:
        应用 LoRA 后的模型
    """
    
    if target_modules is None:
        target_modules = ["linear"]
    
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 检查模块名是否匹配目标模块
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 创建新的 LoRALinear 层替换原层
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[1] if '.' in name else name
                
                parent = model.get_submodule(parent_name) if parent_name else model
                
                lora_linear = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    bias=module.bias is not None
                )
                
                # 复制原始权重
                lora_linear.base_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_linear.base_layer.bias.data = module.bias.data.clone()
                
                # 替换模块
                setattr(parent, child_name, lora_linear)
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型的参数数量
    
    返回:
        (总参数数，可训练参数数)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============ 使用示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("LoRA 核心实现演示")
    print("=" * 60)
    
    # 1. 创建一个简单的线性层
    print("\n1. 创建基础线性层")
    base_linear = nn.Linear(512, 1024)
    total, trainable = count_parameters(base_linear)
    print(f"   总参数：{total:,} | 可训练：{trainable:,}")
    
    # 2. 创建 LoRA 线性层
    print("\n2. 创建 LoRA 线性层 (r=8, alpha=16)")
    lora_linear = LoRALinear(
        in_features=512,
        out_features=1024,
        r=8,
        alpha=16,
        dropout=0.05
    )
    total, trainable = count_parameters(lora_linear)
    print(f"   总参数：{total:,} | 可训练：{trainable:,}")
    print(f"   参数减少比例：{(1 - trainable/total)*100:.2f}%")
    
    # 3. 测试前向传播
    print("\n3. 测试前向传播")
    x = torch.randn(4, 512)  # batch_size=4, features=512
    output = lora_linear(x)
    print(f"   输入形状：{x.shape}")
    print(f"   输出形状：{output.shape}")
    
    # 4. 测试权重合并
    print("\n4. 测试权重合并")
    delta_weight = lora_linear.lora_layer.merge_weights()
    print(f"   增量权重形状：{delta_weight.shape}")
    print(f"   合并后不增加推理延迟 ✓")
    
    # 5. 不同 r 值的参数对比
    print("\n5. 不同 r 值的参数量对比")
    for r in [4, 8, 16, 32, 64]:
        lora = LoRALinear(512, 1024, r=r, alpha=2*r)
        _, trainable = count_parameters(lora)
        print(f"   r={r:2d}: {trainable:>6,} 可训练参数")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
