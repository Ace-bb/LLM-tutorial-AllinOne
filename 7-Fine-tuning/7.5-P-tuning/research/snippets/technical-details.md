# P-tuning 技术原理详解

> 核心组件：虚拟词元 + 提示词编码器（LSTM+MLP）

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    输入序列                              │
│  [虚拟词元_1, ..., 虚拟词元_n, 实际输入文本]              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  提示词编码器                            │
│              LSTM + MLP                                  │
│  输入：虚拟词元索引                                      │
│  输出：连续向量表示（维度=模型嵌入维度）                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                嵌入组合层                                │
│  prompt_embeddings + input_embeddings                    │
│  torch.cat([prompt_embeds, input_embeds], dim=1)         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│           冻结的预训练语言模型                           │
│  GPT-2 / BERT / 其他 Transformer                         │
│  所有参数 requires_grad=False                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    输出层                                │
│  仅基于实际输入部分计算损失                              │
└─────────────────────────────────────────────────────────┘
```

---

## 二、核心组件详解

### 2.1 虚拟词元（Virtual Tokens）

**定义：**
- 不是词汇表中的真实 token
- 是可训练的连续向量（continuous vectors）
- 维度与预训练模型的嵌入维度一致
- 随机初始化或通过启发式方法初始化

**数学表示：**
```
E_virtual ∈ R^(n_prompt × d_model)
```
其中：
- `n_prompt`：虚拟词元数量（通常 50-100）
- `d_model`：预训练模型嵌入维度（如 GPT-2 为 768）

**关键特性：**
1. **可训练：** 通过反向传播更新
2. **连续：** 相比离散 token，可微分优化
3. **灵活：** 可放在输入序列任意位置（前缀、中间、后缀）

**代码实现：**
```python
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim, num_virtual_tokens):
        super().__init__()
        # 虚拟词元嵌入（可训练参数）
        self.virtual_tokens = nn.Embedding(num_virtual_tokens, embed_dim)
        # 初始化：随机或基于真实 token
        nn.init.normal_(self.virtual_tokens.weight, std=0.02)
    
    def forward(self, batch_size):
        # 获取虚拟词元嵌入 [batch_size, n_prompt, embed_dim]
        token_embeds = self.virtual_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return token_embeds
```

---

### 2.2 提示词编码器（Prompt Encoder）

**设计动机：**
> 由于预训练后的嵌入层参数通常是高度离散的，如果随机初始化虚拟词元，模型容易陷入局部最优解。插入的虚拟词元之间应该存在某种关联性。

**架构：** LSTM + MLP

#### LSTM 层

**作用：**
- 捕捉虚拟词元之间的依赖关系
- 重参数化（reparameterization），加速训练收敛
- 双向或单向 LSTM

**参数配置：**
```python
self.lstm = nn.LSTM(
    input_size=embed_dim,      # 输入维度 = 模型嵌入维度
    hidden_size=512,           # 隐藏层维度（固定）
    num_layers=1,              # 单层
    batch_first=True,          # 输入格式 [batch, seq, dim]
    bidirectional=True         # 双向（可选）
)
```

**输出：**
```
LSTM 输出：[batch_size, n_prompt, hidden_size * num_directions]
```

#### MLP 层

**作用：**
- 将 LSTM 输出映射到模型嵌入空间
- 非线性变换，增强表达能力

**参数配置：**
```python
self.mlp = nn.Sequential(
    nn.Linear(hidden_size * 2, hidden_size),  # 双向 LSTM 输出维度×2
    nn.ReLU(),
    nn.Linear(hidden_size, embed_dim)          # 映射回模型嵌入维度
)
```

**数学公式：**
```
h_i = MLP([LSTM(h_0:i) : LSTM(h_i:m)])
```
其中 `:` 表示拼接，`h_0:i` 和 `h_i:m` 分别是双向 LSTM 的前向和后向输出。

**完整编码器实现：**
```python
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim=512, num_virtual_tokens=50):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        
        # 虚拟词元嵌入
        self.virtual_tokens = nn.Embedding(num_virtual_tokens, embed_dim)
        
        # LSTM 编码器
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        
        # MLP 映射
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, batch_size):
        # 获取虚拟词元嵌入
        token_embeds = self.virtual_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # LSTM 编码
        lstm_out, _ = self.lstm(token_embeds)  # [batch, n_prompt, hidden*2]
        
        # MLP 映射到嵌入空间
        prompt_embeds = self.mlp(lstm_out)     # [batch, n_prompt, embed_dim]
        
        return prompt_embeds
```

---

### 2.3 嵌入组合（Embedding Combination）

**目标：** 将提示嵌入和输入嵌入拼接，作为模型输入。

**代码实现：**
```python
def forward(self, input_ids, attention_mask, labels=None):
    batch_size = input_ids.shape[0]
    
    # 1. 获取提示嵌入
    prompt_embeds = self.prompt_encoder(batch_size)  # [batch, n_prompt, embed_dim]
    
    # 2. 获取输入嵌入
    input_embeds = self.model.get_input_embeddings()(input_ids)  # [batch, n_input, embed_dim]
    
    # 3. 拼接提示和输入
    combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)  # [batch, n_prompt+n_input, embed_dim]
    
    # 4. 调整 attention_mask
    prompt_attention_mask = torch.ones(
        (batch_size, self.num_virtual_tokens), 
        device=input_ids.device
    )
    combined_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
    
    # 5. 通过模型（使用 inputs_embeds 而非 input_ids）
    outputs = self.model(
        inputs_embeds=combined_embeds,
        attention_mask=combined_attention_mask,
        labels=labels
    )
    
    return outputs
```

**关键点：**
- 使用 `inputs_embeds` 参数而非 `input_ids`，因为输入包含虚拟词元
- attention_mask 需要扩展以包含虚拟词元部分
- 虚拟词元的 attention_mask 设为 1（可见）

---

## 三、训练流程

### 3.1 参数冻结

**核心原则：** 仅训练提示编码器参数，冻结预训练模型。

```python
# 冻结主模型参数
for param in model.model.parameters():
    param.requires_grad = False

# 仅提示编码器可训练
optimizer = torch.optim.AdamW(
    model.prompt_encoder.parameters(),  # 仅优化编码器参数
    lr=1e-3
)
```

**参数量对比：**
| 组件 | 参数量 | 可训练 |
|------|--------|--------|
| 预训练模型（如 GPT-2） | 1.5B | ❌ |
| 虚拟词元嵌入 | 50 × 768 = 38K | ✅ |
| LSTM | 2 × 768 × 512 × 4 = 3.1M | ✅ |
| MLP | 1024 × 512 + 512 × 768 = 917K | ✅ |
| **总计** | **~1.5B** | **~4M (0.27%)** |

---

### 3.2 训练循环

```python
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # 计算损失（仅基于实际输入部分）
        loss = outputs.loss
        
        # 反向传播（仅更新提示编码器）
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 日志记录
        if step % log_interval == 0:
            print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")
```

**关键点：**
- 损失计算仅基于实际输入部分（排除虚拟词元）
- 梯度仅流向提示编码器
- 预训练模型梯度为 0

---

### 3.3 超参数建议

| 超参数 | 推荐值 | 说明 |
|-------|-------|------|
| **虚拟词元数量** | 50 | 默认值，简单任务 20-30，复杂任务 80-100 |
| **LSTM 隐藏层** | 512 | 固定值，足够捕捉依赖 |
| **学习率** | 1e-3 | 比全量微调大（仅训练少量参数） |
| **批大小** | 16-32 | 根据显存调整 |
| **训练轮数** | 50-200 | 全监督 50-100，少样本 100-200 |
| **优化器** | AdamW | weight_decay=0.01 |
| **虚拟词元初始化** | 正态分布 N(0, 0.02) | 或基于真实 token 嵌入 |

---

## 四、数学原理

### 4.1 目标函数

```
min_θ L(f(x; θ_fixed, θ_prompt), y)
```

其中：
- `θ_fixed`：冻结的模型参数（梯度为 0）
- `θ_prompt`：提示编码器参数（可训练）
- `x`：输入序列
- `y`：标签

### 4.2 梯度流

```
∂L/∂θ_prompt = ∂L/∂output × ∂output/∂θ_prompt
∂L/∂θ_fixed = 0  （参数冻结）
```

**关键：** 梯度仅通过提示编码器反向传播，不更新预训练模型。

### 4.3 前向传播公式

```
E_combined = [E_prompt; E_input]
H = Transformer(E_combined)
P(y|x) = Softmax(H[last_token])
```

其中 `;` 表示拼接。

---

## 五、与相关方法对比

### 5.1 vs 直接学习（Prompt Tuning）

| 维度 | P-tuning | Prompt Tuning |
|------|---------|---------------|
| 编码器 | LSTM+MLP | 无 |
| 虚拟词元关系 | 建模依赖 | 独立学习 |
| 收敛速度 | 快（小模型） | 较慢 |
| 参数量 | 稍多 | 最少 |
| 实现复杂度 | 简单 | 最简单 |

### 5.2 vs Prefix-Tuning

| 维度 | P-tuning | Prefix-Tuning |
|------|---------|---------------|
| 提示位置 | 输入层 | 每层 Transformer |
| 编码器 | LSTM+MLP | 无（直接学习） |
| 参数量 | ~0.01% | ~0.1% |
| 适用任务 | NLU | 生成 |

---

*片段整理时间：2026-03-16*
