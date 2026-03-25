# P-tuning 与 P-tuning V2 对比

> 关键区别：**本文讲解的是原始 P-tuning（2021），不是 P-tuning V2（2022）**

---

## 核心区别速查表

| 对比维度 | P-tuning（原始） | P-tuning V2 |
|---------|----------------|-------------|
| **论文** | GPT Understands, Too (2021) | P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning (2022) |
| **arXiv** | 2103.10385 | 2110.07602 |
| **发表 venue** | ACL 2021 | ACL 2022 |
| **作者** | Liu et al.（清华大学） | Liu et al.（清华大学） |
| **提示位置** | 仅输入层 | **每一层**Transformer |
| **技术名称** | Prompt Tuning | **Deep** Prompt Tuning |
| **编码器** | LSTM+MLP | 可选（通常简化为直接学习） |
| **可训练参数** | ~0.01% | ~0.1%-3% |
| **适用模型** | GPT 类自回归模型 | BERT、RoBERTa、GLM 等双向模型 |
| **适用任务** | NLU、知识探测 | NLU、**序列标注**、生成 |
| **性能声称** | 部分任务接近全量微调 | **与全量微调相当** |
| **实现复杂度** | 简单 | 较复杂 |
| **本文范围** | ✅ **本文讲解** | ❌ 不涵盖 |

---

## 详细对比

### 1. 提示添加位置

**原始 P-tuning：**
```
[虚拟词元_1, ..., 虚拟词元_n, 实际输入] → 冻结的预训练模型 → 输出
                    ↓
              仅在输入层添加
```

**P-tuning V2：**
```
输入层：    [虚拟词元，实际输入]
            ↓
第 1 层：    [虚拟词元，隐藏状态]
            ↓
第 2 层：    [虚拟词元，隐藏状态]
            ↓
...         ...
第 N 层：    [虚拟词元，隐藏状态]
            ↓
           输出
```

**关键差异：** V2 在每一层都添加提示，大幅增加可训练参数容量。

---

### 2. 参数量对比

**原始 P-tuning：**
- 虚拟词元数量：50-100
- 编码器参数：LSTM(512) + MLP
- 总参数量：约 0.01% 模型参数

**P-tuning V2：**
- 每层虚拟词元：5-20
- 层数：12-24 层（根据模型）
- 总参数量：约 0.1%-3% 模型参数

**示例计算（BERT-large）：**
- 原始 P-tuning：~10K 参数
- P-tuning V2：~100K-300K 参数
- 全量微调：~340M 参数

---

### 3. 适用任务范围

**原始 P-tuning 擅长：**
- ✅ 文本分类（情感分析、主题分类）
- ✅ 自然语言推理（NLI）
- ✅ 知识探测（LAMA）
- ✅ 小样本学习（Few-shot）
- ❌ 序列标注（NER、SRL）效果有限
- ❌ 生成任务不如 Prefix-Tuning

**P-tuning V2 扩展：**
- ✅ 以上所有任务
- ✅ **序列标注**（NER、SRL）
- ✅ 问答任务（SQuAD）
- ✅ 更多 NLU 任务

**关键改进来源：** 深层提示（deep prompt）提供更多可训练容量，能处理更复杂任务。

---

### 4. 性能对比

**官方实验结果（来自 P-tuning V2 论文）：**

| 任务 | 全量微调 | P-tuning（v1） | P-tuning V2 |
|------|---------|---------------|-------------|
| **BoolQ** | 86.5 | - | 84.0 |
| **COPA** | 94.0 | - | 92.0 |
| **RTE** | 85.0 | - | 86.6 |
| **WiC** | 76.0 | - | 73.7 |
| **CoNLL03 (NER)** | 93.5 | 无效 | 91.8 |
| **SQuAD 1.1** | 91.0/94.5 | - | 88.1/94.2 |

**关键发现：**
- P-tuning V2 在序列标注任务上首次达到接近全量微调的性能
- 原始 P-tuning 在 NER 等任务上"有效性未得到验证"
- V2 在小模型（BERT-base）上表现更好

---

### 5. 实现复杂度

**原始 P-tuning 实现：**
```python
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_virtual_tokens):
        super().__init__()
        self.virtual_tokens = nn.Embedding(num_virtual_tokens, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, batch_size):
        token_embeds = self.virtual_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        lstm_out, _ = self.lstm(token_embeds)
        prompt_embeds = self.mlp(lstm_out)
        return prompt_embeds

# 仅需在输入层拼接提示
prompt_embeds = self.prompt_encoder(batch_size)
input_embeds = model.get_input_embeddings()(input_ids)
combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
```

**P-tuning V2 实现（简化示意）：**
```python
# 需要为每一层准备提示
self.prompts = nn.ModuleList([
    nn.Embedding(num_virtual_tokens, embed_dim) 
    for _ in range(num_layers)
])

def forward(self, hidden_states):
    # 每一层都要拼接提示
    for layer_idx, layer in enumerate(self.model.layers):
        prompt_embeds = self.prompts[layer_idx].weight.unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = torch.cat([prompt_embeds, hidden_states], dim=1)
        hidden_states = layer(hidden_states)
    return hidden_states
```

**复杂度对比：**
- 原始 P-tuning：单次拼接，简单
- V2：每层拼接，需要修改模型前向传播逻辑

---

### 6. 官方代码仓库

**原始 P-tuning：**
- 仓库：https://github.com/THUDM/P-tuning
- 说明：Codes and datasets for paper "GPT understands, too"
- 数据集：LAMA, FewGLUE_32dev

**P-tuning V2：**
- 仓库：https://github.com/THUDM/P-tuning-v2
- 说明：An optimized deep prompt tuning strategy
- 数据集：SuperGLUE, SQuAD, NER, SRL

**仓库明确说明：**
> Find our previous version P-tuning v1 for knowledge probing and few-shot SuperGLUE.

---

### 7. 如何选择

**选择原始 P-tuning 如果：**
- ✅ 需要极简实现
- ✅ 仅处理 NLU 分类任务
- ✅ 模型规模较大（>1B 参数）
- ✅ 资源极度受限（显存<8GB）

**选择 P-tuning V2 如果：**
- ✅ 需要处理序列标注任务
- ✅ 模型规模较小（<1B 参数）
- ✅ 追求最佳性能
- ✅ 有足够计算资源

**选择其他方法（LoRA）如果：**
- ✅ 生成任务（文本生成、对话）
- ✅ 需要更广泛支持
- ✅ 生产环境部署

---

## 本文范围声明

**本文《P-tuning 技术详解》讲解的是：**
- ✅ **原始 P-tuning**（arxiv:2103.10385, 2021）
- ✅ 输入层添加虚拟词元
- ✅ LSTM+MLP 编码器架构
- ✅ NLU 任务应用

**本文不涵盖：**
- ❌ P-tuning V2（深层提示，每层添加）
- ❌ P-tuning V2 的序列标注任务扩展
- ❌ P-tuning V2 的复杂实现细节

**如需了解 P-tuning V2，请参考：**
- 论文：arxiv:2110.07602
- 代码：https://github.com/THUDM/P-tuning-v2

---

*片段整理时间：2026-03-16*
