# LoRA 微调参考资料

> 本文档整理了 LoRA（Low-Rank Adaptation）微调技术的核心参考资料，包括原始论文、官方文档、调参最佳实践、性能对比数据和实战经验。

---

## 目录

1. [LoRA 原始论文核心数据](#1-lora-原始论文核心数据)
2. [HuggingFace PEFT 库官方文档](#2-huggingface-peft-库官方文档)
3. [LoRA 调参最佳实践](#3-lora-调参最佳实践)
4. [性能对比数据](#4-性能对比数据)
5. [实战经验总结](#5-实战经验总结)
6. [LoRA 变体技术](#6-lora-变体技术)
7. [应用案例](#7-应用案例)

---

## 1. LoRA 原始论文核心数据

### 论文信息

- **标题**: LoRA: Low-Rank Adaptation of Large Language Models
- **作者**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **机构**: 微软研究院
- **发表**: ICLR 2022
- **链接**: https://arxiv.org/abs/2106.09685
- **代码仓库**: https://github.com/microsoft/LoRA

### 核心方法

LoRA 通过冻结预训练模型权重，学习低秩分解矩阵来实现参数高效微调：

```
W_updated = W + ΔW = W + BA
```

其中 B ∈ R^(d×r), A ∈ R^(r×k)，秩 r << min(d, k)

### 核心实验结果

#### GLUE 基准测试结果（RoBERTa 和 DeBERTa）

| 模型 | 方法 | 可训练参数 | MNLI | SST2 | MRPC | CoLA | QNLI | QQP | RTE | STSB | 平均 |
|------|------|-----------|------|------|------|------|------|-----|-----|------|------|
| RoBERTa base | 全量微调 | 125M | 87.6 | 94.8 | 90.2 | 63.6 | 92.8 | 91.9 | 78.7 | 91.2 | 86.40 |
| RoBERTa base | LoRA | **0.8M** | 87.5±.3 | 95.1±.2 | 89.7±.7 | 63.4±1.2 | 93.3±.3 | 90.8±.1 | 86.6±.7 | 91.5±.2 | **87.24** |
| DeBERTa XXL | 全量微调 | 1.5B | 91.7/91.9 | 97.2 | 92.0 | 72.0 | 96.0 | 92.7 | 93.9 | 92.9/92.6 | 91.06 |
| DeBERTa XXL | LoRA | **4.7M** | 91.9±.1/91.9±.2 | 96.9±.2 | 92.6±.6 | 72.4±1.1 | 96.0±.1 | 92.9±.1 | 94.9±.4 | 93.0±.2/92.9±.3 | **91.32** |

#### GPT-2 文本生成结果

| 模型 | 方法 | 可训练参数 | E2E (BLEU) | DART (BLEU) | WebNLG (BLEU-U/S/A) |
|------|------|-----------|------------|-------------|---------------------|
| GPT-2 Medium | 全量微调 | 354.92M | 68.2 | 46.0 | 30.4/63.2/47.6 |
| GPT-2 Medium | Adapter | 0.37M | 66.3 | 42.4 | 45.1/54.5/50.2 |
| GPT-2 Medium | Prefix | 0.35M | 69.7 | 45.7 | 44.1/63.1/54.4 |
| GPT-2 Medium | **LoRA** | **0.35M** | **70.4±.1** | **47.1±.2** | **46.7±.4/62.1±.2/55.3±.2** |
| GPT-2 Large | 全量微调 | 774.03M | 68.5 | 46.5 | 41.7/64.6/54.2 |
| GPT-2 Large | **LoRA** | **0.77M** | **70.4±.1** | **47.5±.1** | **48.4±.3/64.0±.3/57.0±.1** |

### 关键发现

1. **性能相当或更优**: LoRA 在 GLUE 基准上达到或超过全量微调的性能
2. **参数效率**: 仅训练 0.1%-1% 的参数
3. **无推理延迟**: 训练完成后可合并权重，不增加推理延迟
4. **任务切换高效**: 不同任务只需保存小的 LoRA 权重（MB 级别）

---

## 2. HuggingFace PEFT 库官方文档

### PEFT 库信息

- **GitHub**: https://github.com/huggingface/peft
- **文档**: https://huggingface.co/docs/peft
- **安装**: `pip install peft`

### LoraConfig 核心参数

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,                          # LoRA 秩
    lora_alpha=32,                 # 缩放因子 α
    target_modules=["q_proj", "v_proj"],  # 目标模块
    lora_dropout=0.05,             # Dropout 比例
    bias="none",                   # 是否训练 bias
    task_type=TaskType.CAUSAL_LM,  # 任务类型
    inference_mode=False,          # 是否推理模式
    modules_to_save=None,          # 额外需要训练的模块
    init_lora_weights="gaussian",  # 初始化方法
)
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `r` | int | 8 | LoRA 秩，控制可训练参数量 |
| `lora_alpha` | int | 16 | 缩放因子，实际缩放为 α/r |
| `target_modules` | list[str] | None | 应用 LoRA 的模块名列表 |
| `lora_dropout` | float | 0.0 | LoRA 层的 dropout 概率 |
| `bias` | str | "none" | "none"/"all"/"lora_only" |
| `task_type` | TaskType | None | 任务类型（CAUSAL_LM/SEQ_2_SEQ_LM 等） |
| `inference_mode` | bool | False | 推理模式下不训练 |
| `modules_to_save` | list[str] | None | 除了 LoRA 外还需要训练的模块 |
| `init_lora_weights` | str/bool | "gaussian" | 权重初始化方法 |

### 使用示例

#### 基础用法

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,  # QLoRA
    device_map="auto"
)

# 准备模型用于 k-bit 训练
model = prepare_model_for_kbit_training(model)

# 配置 LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用 LoRA
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

#### 保存和加载

```python
# 保存 LoRA 权重
model.save_pretrained("./lora-weights")

# 加载 LoRA 权重
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True
)
model = PeftModel.from_pretrained(base_model, "./lora-weights")
```

---

## 3. LoRA 调参最佳实践

### 3.1 秩 r 的选择建议

| 模型规模 | 推荐 r 值 | 适用场景 |
|----------|----------|----------|
| 小型模型 (<1B) | 4-8 | 简单任务，数据量少 |
| 中型模型 (1B-7B) | 8-16 | 通用场景 |
| 大型模型 (7B-13B) | 16-32 | 复杂任务，数据量大 |
| 超大型模型 (>13B) | 32-64 | 专业领域微调 |

**来源**: Sebastian Raschka 实验总结 https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

#### 关键发现

- **r=8** 是良好的起点，适用于大多数场景
- **r 增加** 会提升性能但增加显存占用和训练时间
- **r=256** 时 Adam vs SGD 内存差异显著（17.86GB vs 14.46GB）
- **最佳 r 值** 需要通过实验验证，建议从 r=8 开始尝试

### 3.2 学习率推荐范围

| 方法 | 推荐学习率 | 说明 |
|------|-----------|------|
| LoRA (全精度) | 1e-4 ~ 2e-4 | 标准设置 |
| LoRA (8-bit) | 1e-3 ~ 2e-3 | 可提高 10 倍 |
| LoRA (4-bit QLoRA) | 1e-4 ~ 1e-3 | 需要实验调优 |
| LoRA+ | A: 1e-3, B: 1e-4 | A 矩阵学习率是 B 的 10 倍 |

**来源**: Phil Schmid 教程 https://www.philschmid.de/fine-tune-flan-t5-peft

#### 学习率调度器

- **Cosine Annealing**: 对 SGD 效果显著，对 Adam/AdamW 影响较小
- **Linear Warmup**: 推荐 warmup 比例 0.03-0.1
- **固定学习率**: 简单场景可用

### 3.3 缩放因子 α 设置

**推荐规则**: `alpha = 2 * r`

| r 值 | 推荐 α | 缩放因子 (α/r) |
|-----|-------|---------------|
| 8 | 16 | 2.0 |
| 16 | 32 | 2.0 |
| 32 | 64 | 2.0 |
| 64 | 128 | 2.0 |

**来源**: Sebastian Raschka 实验

#### 原理

```python
scaling = alpha / r
weight += (lora_B @ lora_A) * scaling
```

- α/r=2 是经验最佳值
- 保持 α/r 比例恒定，改变 r 时同步调整 α

### 3.4 Dropout 比例

| 场景 | 推荐 dropout | 说明 |
|------|------------|------|
| 小数据集 (<1k) | 0.1-0.2 | 防止过拟合 |
| 中等数据集 (1k-10k) | 0.05-0.1 | 平衡 |
| 大数据集 (>10k) | 0.0-0.05 | 低 dropout |
| 指令微调 | 0.05 | 通用推荐值 |

### 3.5 Target Modules 选择

#### 推荐配置（按性能排序）

1. **全部线性层** (最佳性能)
   ```python
   target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]
   ```
   - 可训练参数增加 5 倍
   - 性能提升明显

2. **注意力层** (推荐)
   ```python
   target_modules=["q_proj", "v_proj"]
   ```
   - 原始论文推荐
   - 性价比最高

3. **仅注意力层** (基础)
   ```python
   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
   ```

**来源**: Microsoft LoRA 官方仓库

### 3.6 完整配置示例

```python
# 7B 模型通用配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 13B+ 模型高性能配置
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 小数据量防过拟合配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## 4. 性能对比数据

### 4.1 LoRA vs 全量微调

#### FLAN-T5 XXL (11B) 对比

| 指标 | LoRA + 8-bit | 全量微调 | 对比 |
|------|-------------|---------|------|
| 可训练参数 | 18.9M (0.17%) | 11.2B (100%) | **590x 减少** |
| 显存需求 | 18GB (单卡 A10G) | 320GB (8x A100 40GB) | **17.8x 减少** |
| 训练时间 | ~10 小时 | ~10 小时 | 相当 |
| 成本 | $13.22 | $322 | **24x 降低** |
| LoRA checkpoint | 84MB | 44GB | **524x 减小** |
| ROUGE-1 | 50.39% | 47.23% (T5-base) | **+3.16%** |

**来源**: Phil Schmid 教程 https://www.philschmid.de/fine-tune-flan-t5-peft

#### Llama 2 7B 对比（Sebastian Raschka 实验）

| 方法 | 显存 | 训练时间 | 性能 |
|------|------|---------|------|
| 全量微调 | ~80GB | - | 基准 |
| LoRA (r=8) | 14.18GB | 1.85h | 相当 |
| QLoRA (4-bit) | 14.18GB | 2.79h (+39%) | 相当 |

### 4.2 显存节省量化数据

#### 7B 模型不同配置显存对比

| 配置 | 显存占用 | 相比全量节省 |
|------|---------|-------------|
| 全量微调 (FP16) | ~80GB | - |
| LoRA (r=8, FP16) | 14.18GB | **82%** |
| LoRA (r=16, FP16) | 16.62GB | **79%** |
| LoRA (r=256, FP16) | 17.86GB | **78%** |
| QLoRA (4-bit) | 14.18GB | **82%** |
| LoRA (r=256, SGD) | 14.46GB | **82%** |

**来源**: Sebastian Raschka 实验

#### 不同 r 值可训练参数量（Llama 2 7B）

| r 值 | 可训练参数 | 占总参数比例 |
|------|-----------|-------------|
| 8 | 4.2M | 0.06% |
| 16 | 8.4M | 0.12% |
| 32 | 16.8M | 0.25% |
| 64 | 33.6M | 0.50% |
| 256 | 134.2M | 2.0% |

### 4.3 训练时间对比

#### FLAN-T5 XXL 训练时间

| 方法 | 硬件 | 时间 | 成本 |
|------|------|------|------|
| LoRA + 8-bit | 1x A10G | 10h 36m | $13.22 |
| 全量微调 | 8x A100 40GB | 10h | $322 |

#### Llama 2 7B 训练时间（Alpaca 50k）

| 方法 | 硬件 | 时间 |
|------|------|------|
| LoRA (r=8) | 1x A10G | 1.85h |
| QLoRA (4-bit) | 1x A10G | 2.79h (+39%) |

### 4.4 LoRA+ 性能提升

| 指标 | LoRA | LoRA+ | 提升 |
|------|------|-------|------|
| 性能 | 基准 | +1-2% | **1-2%** |
| 训练速度 | 基准 | 2x | **2 倍加速** |
| 计算成本 | 相同 | 相同 | 无增加 |

**来源**: LoRA+ 论文 https://arxiv.org/abs/2402.12354

---

## 5. 实战经验总结

### 5.1 社区常见问题和解决方案

#### Q1: 如何选择合适的秩 r？

**答案**: 
- 从 r=8 开始实验
- 如果欠拟合，增加到 16 或 32
- 如果过拟合，减少到 4 或保持 8 并增加 dropout
- 大数据集可以用更大的 r（32-64）

#### Q2: LoRA 是否适用于领域适应？

**答案**: 
- 是的，LoRA 特别适合领域适应
- 在专业领域（医疗、法律、金融）表现良好
- 建议使用较大的 r（16-32）和全量 target_modules

#### Q3: 如何避免过拟合？

**答案**:
1. 增加 dropout (0.1-0.2)
2. 减少 r 值 (4-8)
3. 减少训练轮次 (1-2 epochs)
4. 使用早停（early stopping）
5. 增加数据多样性

**来源**: Sebastian Raschka

#### Q4: 多轮训练（multi-epoch）是否有益？

**答案**: **通常无益，甚至有害**

- 静态数据集（如 Alpaca 50k）多轮训练会导致过拟合
- 观察到 2 epochs 比 1 epoch 性能下降
- 建议：1 epoch 或更少

#### Q5: 优化器选择重要吗？

**答案**: **不太重要**

- AdamW、SGD+scheduler、AdamW+scheduler 差异很小
- LoRA 参数少，优化器状态内存占用小
- r 较大时（256+）SGD 可节省内存（17.86GB → 14.46GB）

#### Q6: LoRA 需要应用到所有层吗？

**答案**: **推荐应用到所有线性层**

- 仅 q、v 矩阵：基准性能
- 所有注意力层（q、k、v、o）：性能提升
- 所有线性层（包括 MLP）：最佳性能，但显存增加

实验数据（Llama 2 7B）：
- q、v 矩阵：14.18GB，基准性能
- 所有层：16.62GB，性能明显提升

### 5.2 踩坑记录

#### 坑 1: 忘记设置 `merge_weights=False`

**问题**: 训练时 model.eval() 会合并权重，导致无法继续训练

**解决**: 
```python
# 确保训练模式下不自动合并
model.train()
# 或在配置中设置
config = LoraConfig(..., merge_weights=False)
```

#### 坑 2: target_modules 名称错误

**问题**: 不同模型架构的模块名不同

**解决**: 先打印模型结构确认模块名
```python
print(model)
# 常见模块名：
# Llama: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
# GPT-2: c_attn, c_proj, c_fc
# T5: q, v, o, k
```

#### 坑 3: 加载权重时 strict=True

**问题**: 加载 LoRA 权重时报错

**解决**:
```python
# 错误
model.load_state_dict(torch.load('lora.ckpt'))

# 正确
model.load_state_dict(torch.load('lora.ckpt'), strict=False)
```

#### 坑 4: 量化后未准备模型

**问题**: QLoRA 训练时未调用 `prepare_model_for_kbit_training`

**解决**:
```python
model = AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)
model = prepare_model_for_kbit_training(model)  # 必须调用
model = get_peft_model(model, config)
```

#### 坑 5: 学习率设置过高

**问题**: 使用全量微调的学习率（1e-5）导致训练不稳定

**解决**: LoRA 使用更高的学习率（1e-4 ~ 2e-4），8-bit 可用 1e-3

### 5.3 优化技巧

#### 技巧 1: 使用 FlashAttention-2

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    use_flash_attention_2=True,  # 加速训练
    load_in_4bit=True,
    device_map="auto"
)
```

**效果**: 训练速度提升 20-30%

#### 技巧 2: 梯度累积

```python
training_args = TrainingArguments(
    ...,
    gradient_accumulation_steps=4,  # 等效 batch_size x4
    per_device_train_batch_size=2,
)
```

**效果**: 在不增加显存的情况下增大有效 batch_size

#### 技巧 3: 混合精度训练

```python
training_args = TrainingArguments(
    ...,
    fp16=True,  # 或 bf16=True
)
```

**效果**: 显存节省约 50%，速度提升

#### 技巧 4: 使用 Liger Kernel

```python
# LLaMA-Factory 配置
enable_liger_kernel: true
```

**效果**: 训练效率显著提升

#### 技巧 5: 数据集打包（Packed Dataset）

```python
# LLaMA-Factory 配置
neat_packing: true
```

**效果**: 减少 padding，提升训练效率

---

## 6. LoRA 变体技术

### 6.1 QLoRA (Quantized LoRA)

**论文**: https://arxiv.org/abs/2305.14314

**核心改进**:
- 4-bit 量化预训练权重
- 分页优化器（Paged Optimizers）处理内存峰值
- 保持 LoRA 参数为 FP16

**优势**:
- 显存节省 33%（相比 LoRA）
- 性能几乎无损
- 可在单卡 24GB 微调 65B 模型

**配置**:
```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    device_map="auto"
)
```

### 6.2 LoRA+

**论文**: https://arxiv.org/abs/2402.12354

**核心改进**:
- A 和 B 矩阵使用不同学习率
- 推荐比例：lr_A = 16 * lr_B

**优势**:
- 性能提升 1-2%
- 训练速度提升 2 倍
- 无额外计算成本

**配置**:
```python
# 使用 PEFT 0.10.0+
from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=32,
    use_rslora=True,  # 启用 Rank-Stabilized LoRA
    ...
)

# 或使用 LoRA+ 优化器
from torch.optim import AdamW

optimizer = AdamW([
    {"params": model.lora_A.parameters(), "lr": 1e-3},
    {"params": model.lora_B.parameters(), "lr": 1e-4},
])
```

### 6.3 AdaLoRA (Adaptive LoRA)

**论文**: https://arxiv.org/abs/2303.10512

**核心改进**:
- 自适应分配参数预算
- 根据重要性分数剪枝奇异值
- SVD 形式参数化增量更新

**优势**:
- 低预算设置下表现更好
- 自动识别重要权重矩阵
- 无需手动调 r 值

**配置**:
```python
from peft import AdaLoraConfig

config = AdaLoraConfig(
    init_r=12,
    target_r=8,
    beta1=0.85,
    beta2=0.999,
    tinit=200,
    tfinal=1000,
    ...
)
```

### 6.4 QA-LoRA (Quantization-Aware LoRA)

**论文**: https://arxiv.org/abs/2309.14717

**核心改进**:
- 量化感知训练
- 组算子增加量化自由度
- 训练后直接集成量化模型

**优势**:
- 训练时 INT4 量化减少显存
- 训练后无损集成
- 适合边缘设备部署

### 6.5 其他变体

| 变体 | 特点 | 适用场景 |
|------|------|----------|
| **DoRA** | 权重分解为幅度和方向 | 需要更精细控制 |
| **PiSSA** | 主奇异分量初始化 | 加速收敛 |
| **LongLoRA** | 支持长上下文 | 长文本任务 |
| **LoftQ** | 量化 + 低秩联合优化 | 极致压缩 |
| **rsLoRA** | 秩稳定缩放 | 大 r 值场景 |

---

## 7. 应用案例

### 7.1 开源项目案例

#### LLaMA-Factory

- **GitHub**: https://github.com/hiyouga/LLaMA-Factory
- **支持模型**: 100+ LLMs & VLMs
- **支持方法**: LoRA, QLoRA, DoRA, LoRA+, AdaLoRA 等
- **特点**: 
  - 统一训练框架
  - 支持 Web UI
  - 多卡分布式训练
  - 多模态支持

**使用示例**:
```bash
# 命令行训练
llamafactory-cli train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --dataset alpaca \
    --output_dir ./saves/lora-llama2
```

#### Unsloth

- **GitHub**: https://github.com/unslothai/unsloth
- **特点**: 
  - 2x 更快训练速度
  - 60% 显存节省
  - 支持 Llama、Mistral、Gemma 等

### 7.2 企业应用案例

#### 案例 1: 医疗诊断（70B 模型）

- **模型**: Llama3.1-70B
- **方法**: LoRA + QLoRA
- **硬件**: 2x 4090 (24GB)
- **数据**: 医疗问答数据集
- **结果**: 达到专业医生水平

**来源**: LLaMA-Factory 博客

#### 案例 2: 自动驾驶（多模态）

- **模型**: Qwen2.5-VL
- **方法**: LoRA 多模态微调
- **场景**: 个人导游助手
- **硬件**: 单卡 A100

**来源**: PAI-ML Gallery

#### 案例 3: 银行文档视觉信息提取

- **公司**: Apoidea Group
- **平台**: Amazon SageMaker HyperPod
- **模型**: 多模态模型
- **方法**: LLaMA-Factory + LoRA
- **场景**: 银行文档信息提取

**来源**: AWS 博客

### 7.3 典型训练配置

#### 7B 模型单卡配置

```yaml
model: meta-llama/Llama-2-7b-hf
method: QLoRA
hardware: 1x RTX 4090 (24GB)
config:
  r: 16
  alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  dropout: 0.05
  learning_rate: 2e-4
  batch_size: 4
  gradient_accumulation: 4
  epochs: 1
  max_seq_length: 2048
estimated_time: ~2 hours
estimated_vram: ~16GB
```

#### 13B 模型多卡配置

```yaml
model: meta-llama/Llama-2-13b-hf
method: LoRA
hardware: 2x A100 (40GB)
config:
  r: 32
  alpha: 64
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]
  dropout: 0.1
  learning_rate: 1e-4
  batch_size: 2
  gradient_accumulation: 8
  epochs: 1
  max_seq_length: 4096
estimated_time: ~4 hours
estimated_vram: ~30GB per GPU
```

---

## 附录

### A. 关键资源链接

| 资源 | 链接 |
|------|------|
| LoRA 原始论文 | https://arxiv.org/abs/2106.09685 |
| LoRA 官方代码 | https://github.com/microsoft/LoRA |
| HuggingFace PEFT | https://github.com/huggingface/peft |
| PEFT 文档 | https://huggingface.co/docs/peft |
| QLoRA 论文 | https://arxiv.org/abs/2305.14314 |
| LoRA+ 论文 | https://arxiv.org/abs/2402.12354 |
| AdaLoRA 论文 | https://arxiv.org/abs/2303.10512 |
| QA-LoRA 论文 | https://arxiv.org/abs/2309.14717 |
| Sebastian Raschka 博客 | https://magazine.sebastianraschka.com |
| Phil Schmid 教程 | https://www.philschmid.de |
| LLaMA-Factory | https://github.com/hiyouga/LLaMA-Factory |

### B. 快速参考卡片

```
LoRA 调参快速参考
=================

秩 r:
  - 起点：r=8
  - 小模型：<8
  - 大模型：16-32
  - 超大数据集：64

学习率:
  - FP16: 1e-4 ~ 2e-4
  - 8-bit: 1e-3 ~ 2e-3
  - 4-bit: 1e-4 ~ 1e-3
  - LoRA+: lr_A = 16 * lr_B

Alpha:
  - 规则：alpha = 2 * r
  - 范围：16-128

Dropout:
  - 小数据集：0.1-0.2
  - 中等数据：0.05-0.1
  - 大数据集：0.0-0.05

Target Modules:
  - 基础：["q_proj", "v_proj"]
  - 推荐：["q_proj", "k_proj", "v_proj", "o_proj"]
  - 最佳：全部线性层

Batch Size:
  - 7B: 2-4
  - 13B: 1-2
  - 70B: 1 (QLoRA)

Epochs:
  - 推荐：1
  - 最大：2-3（防过拟合）
```

---

*文档生成时间：2026-03-16*
*资料来源：上述链接的权威论文、官方文档和技术博客*
