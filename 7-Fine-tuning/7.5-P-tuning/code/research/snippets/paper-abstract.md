# P-tuning 原始论文关键片段

来源：《GPT Understands, Too》(arxiv:2103.10385)

---

## 论文标题与作者

**标题：** GPT Understands, Too

**作者：** Xiao Liu*, Yanan Zheng*, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, Jie Tang

**机构：** 清华大学

**发表时间：** 2021 年 3 月（arXiv 首版）

---

## 摘要（Abstract）

> Prompting a pretrained language model with natural language patterns has been proved effective for natural language understanding (NLU). However, our preliminary study reveals that manual discrete prompts often lead to unstable performance -- e.g., changing a single word in the prompt might result in substantial performance drop. We propose a novel method P-Tuning that employs trainable continuous prompt embeddings in concatenation with discrete prompts. Empirically, P-Tuning not only stabilizes training by minimizing the gap between various discrete prompts, but also improves performance by a sizeable margin on a wide range of NLU tasks including LAMA and SuperGLUE. P-Tuning is generally effective for both frozen and tuned language models, under both the fully-supervised and few-shot settings.

**核心要点：**
1. 手动离散提示不稳定：单个词的改动可能导致性能大幅下降
2. P-tuning 使用可训练的连续提示嵌入
3. 在 LAMA 和 SuperGLUE 等 NLU 任务上取得显著提升
4. 适用于冻结和微调的模型，全监督和少样本设置

---

## 核心方法（Method）

### 虚拟词元（Virtual Tokens）

P-tuning 引入虚拟词元作为可学习的连续提示：

- 不是词汇表中的真实 token
- 是随机初始化的连续向量
- 维度与预训练模型的嵌入维度一致
- 通过反向传播优化

### 提示词编码器（Prompt Encoder）

**架构：** LSTM + MLP

**设计动机：**
> 由于预训练后的嵌入层参数通常是高度离散的，如果随机初始化虚拟词元，模型容易陷入局部最优解。P-tuning 的研究者认为，插入的虚拟词元之间应该存在某种关联性。

**LSTM 作用：**
- 捕捉虚拟词元之间的依赖关系
- 重参数化，加速训练收敛
- 双向或单向 LSTM

**MLP 作用：**
- 将 LSTM 输出映射到模型嵌入空间
- 公式：$h_{i}=\mathrm{MLP}([\mathrm{LSTM}(h_{0:i})\colon\mathrm{LSTM}(h_{i:m})])$

---

## 关键实验结果

### LAMA 知识探测

- P-tuning 在 LAMA benchmark 上显著优于手动提示
- 稳定训练，减少不同提示之间的性能差距
- 在冻结模型和微调模型上都有效

### SuperGLUE

- 在少样本设置（32-dev）下测试
- 多个任务上超越全量微调
- 特别是在 BoolQ、RTE、WiC 等任务

---

## 重要发现

### 模型规模影响

> 当模型的参数规模超过 100 亿时，P-tuning 的效果可以媲美全量微调。但对于参数规模较小的模型，P-tuning 的表现和全参数微调相比差距较大。

### 混合提示策略

P-tuning 支持：
- 纯连续提示（仅虚拟词元）
- 混合提示（虚拟词元 + 离散自然语言 token）
- 锚字符（anchor tokens）提升表现

---

## 局限性（论文中提及）

1. **序列标注任务：** 原始 P-tuning 在序列标注任务上的有效性未得到充分验证
2. **生成任务：** 主要针对 NLU 任务，生成任务效果不如 Prefix-Tuning
3. **超参数敏感：** 虚拟词元数量需要调优

---

## 与相关方法对比

### vs Prefix-Tuning

| 维度 | Prefix-Tuning | P-tuning |
|------|--------------|----------|
| 提示位置 | 每层 Transformer | 输入层 |
| 编码器 | 无 | LSTM+MLP |
| 适用任务 | 生成任务 | NLU 任务 |
| 实现复杂度 | 中等 | 简单 |

### vs Prompt Tuning（Lester et al.）

| 维度 | Prompt Tuning | P-tuning |
|------|--------------|----------|
| 编码器 | 无（直接学习） | LSTM+MLP |
| 收敛速度 | 较慢（小模型） | 较快 |
| 复杂度 | 最简单 | 简单 |

---

## 引用格式

```bibtex
@article{liu2021gpt,
  title={GPT Understands, Too},
  author={Liu, Xiao and Zheng, Yanan and Du, Zhengxiao and Ding, Ming and Qian, Yujie and Yang, Zhilin and Tang, Jie},
  journal={arXiv:2103.10385},
  year={2021}
}
```

---

*片段提取时间：2026-03-16*
