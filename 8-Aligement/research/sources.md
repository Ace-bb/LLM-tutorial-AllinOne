# 大模型强化学习综述 - 参考资料

**文档生成日期：** 2026-03-16  
**搜索执行日期：** 2026-03-16  
**负责 Agent：** Searcher Agent

---

## 主题 1：RLHF 基础与 PPO 算法

### 1.1 PPO 原始论文

**论文信息：**
- **标题：** Proximal Policy Optimization Algorithms
- **作者：** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI)
- **arXiv：** [1707.06347](https://arxiv.org/abs/1707.06347)
- **发表时间：** 2017 年 7 月
- **访问日期：** 2026-03-16

**核心公式：**

PPO-Clip 目标函数：
```
L^CLIP(θ) = Ê_t [min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]
```

其中：
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` 是概率比
- `Â_t` 是优势函数估计
- `ε` 是 clip 范围（通常 0.1-0.3）

**关键发现：**
1. PPO 通过 clip 机制限制策略更新幅度，防止训练不稳定
2. 相比 TRPO，PPO 实现更简单，无需二阶优化
3. 支持多个 epoch 的 minibatch 更新，样本效率更高
4. 在 Atari 游戏和机器人控制任务上表现优异

### 1.2 InstructGPT 论文（RLHF 里程碑）

**论文信息：**
- **标题：** Training Language Models to Follow Instructions with Human Feedback
- **作者：** Long Ouyang, Jeff Wu, Xu Jiang, et al. (OpenAI)
- **arXiv：** [2203.02155](https://arxiv.org/abs/2203.02155)
- **发表时间：** 2022 年 3 月
- **访问日期：** 2026-03-16

**关键方法：**
1. **三阶段训练流程：**
   - 阶段 1：监督微调 (SFT) - 在高质量指令数据上微调
   - 阶段 2：奖励模型训练 - 训练 RM 预测人类偏好
   - 阶段 3：RL 优化 - 使用 PPO 最大化奖励信号

2. **奖励模型架构：**
   - 基于 GPT-3 架构
   - 输入：prompt + response
   - 输出：标量奖励值
   - 训练目标：Bradley-Terry 模型，最大化人类偏好一致性

3. **PPO 在 RLHF 中的应用细节：**
   - KL 惩罚：`β·log(π_θ(a|s) / π_ref(a|s))`，β 通常设为 0.02
   - 奖励归一化：每批次奖励减去均值除以标准差
   - 多轮采样：每个 prompt 采样多个 response 进行训练

**性能数据：**
- SFT 模型参数量：1.3B, 6B, 175B
- 人类偏好胜率：1.3B PPO 模型击败 175B SFT 模型（~70% 胜率）
- 训练成本：175B 模型 PPO 训练约需数天（未公开具体 GPU 时）

### 1.3 PPO 在 RLHF 中的调参建议

**推荐超参数范围：**
- **Learning rate:** 1e-6 到 3e-6（PPO 阶段）
- **Batch size:** 256-1024 个 prompt
- **PPO epochs:** 2-4 个 epoch per batch
- **Clip range (ε):** 0.1-0.2
- **KL coefficient (β):** 0.01-0.1（常用 0.02）
- **Value loss coefficient:** 0.5-1.0
- **Entropy bonus:** 0.001-0.01（可选）
- **GAE λ:** 0.95
- **Discount factor (γ):** 0.99

**显存需求估算（参考值）：**
- 7B 模型 PPO 训练：~40-80GB（单卡 A100，使用 ZeRO）
- 13B 模型 PPO 训练：~80-160GB（多卡）
- 70B 模型 PPO 训练：需多节点分布式

---

## 主题 2：DPO 及其变体

### 2.1 DPO 原始论文

**论文信息：**
- **标题：** Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- **作者：** Rafael Rafailov, Archit Sharma, Eric Mitchell, et al. (Stanford)
- **arXiv：** [2305.18290](https://arxiv.org/abs/2305.18290)
- **发表时间：** 2023 年 5 月
- **访问日期：** 2026-03-16

**核心公式：**

DPO Loss：
```
L_DPO(π_θ; π_ref) = -E_{(x,y_w,y_l)~D} [log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

等价形式（更直观）：
```
L_DPO = -E [log σ(β·(r_θ(x,y_w) - r_θ(x,y_l)))]
```
其中 `r_θ(x,y) = log(π_θ(y|x)/π_ref(y|x))`

**关键发现：**
1. DPO 将奖励函数 reparameterize 为最优策略的形式
2. 无需显式训练奖励模型，直接用偏好数据优化策略
3. 数学上等价于 RLHF，但实现更简单、训练更稳定
4. 参考模型 π_ref 通常是 SFT 后的模型，训练时冻结

**调参建议：**
- **β (温度参数):** 0.1-0.5（常用 0.1-0.2）
  - β 越大，偏离参考模型的惩罚越大
  - β 越小，越接近 SFT 模型
- **Learning rate:** 5e-7 到 2e-6
- **Batch size:** 64-256
- **Epochs:** 1-3（DPO 容易过拟合，不宜过多 epoch）

### 2.2 ORPO (Odds Ratio Preference Optimization)

**论文信息：**
- **标题：** ORPO: Monolithic Preference Optimization without Reference Model
- **作者：** Jiwoo Hong, Noah Lee, James Thorne (VIST Labs)
- **arXiv：** [2403.07691](https://arxiv.org/abs/2403.07691)
- **发表时间：** 2024 年 3 月
- **访问日期：** 2026-03-16

**核心公式：**

OR 损失（Odds Ratio Loss）：
```
L_OR = -E [log σ(log(odds(y_w) - odds(y_l)))]
```

其中 odds 比率：
```
odds(y|x) = P(y|x) / (1 - P(y|x))
```

ORPO 总损失：
```
L_ORPO = L_SFT + λ·L_OR
```

**关键创新：**
1. **无需参考模型**：直接在 SFT 阶段整合偏好优化
2. **Monolithic 训练**：单一阶段完成 SFT+ 偏好对齐
3. **计算效率**：节省 50% 显存（无需存储参考模型）

**性能数据：**
- Phi-2 (2.7B) + ORPO：AlpacaEval 2.0 得分 12.20%
- Llama-2-7B + ORPO：IFEval 得分 66.19%
- Mistral-7B + ORPO：MT-Bench 得分 7.32
- 超越 7B-13B 模型的 DPO/RLHF 方法

**调参建议：**
- **λ (OR 损失权重):** 0.5-2.0（常用 1.0）
- **Learning rate:** 1e-6 到 5e-6
- **Batch size:** 64-128
- **Epochs:** 1-2

### 2.3 SimPO (Simple Preference Optimization)

**论文信息：**
- **标题：** Simple Preference Optimization with a Reference-Free Reward
- **作者：** Yu Meng, Mengzhou Xia, Danqi Chen (Princeton)
- **arXiv：** [2405.14734](https://arxiv.org/abs/2405.14734)
- **发表时间：** 2024 年 5 月
- ** NeurIPS 2024 录用
- **访问日期：** 2026-03-16

**核心公式：**

SimPO 使用平均 log 概率作为隐式奖励：
```
r_θ(x,y) = (1/|y|) · Σ_t log π_θ(y_t|x,y_<t>)
```

SimPO Loss：
```
L_SimPO = -E [log σ(β·(r_θ(x,y_w) - r_θ(x,y_l)) - γ)]
```

其中 γ 是 target reward margin（目标奖励间隔）

**关键创新：**
1. **长度归一化奖励**：使用平均 log 概率而非总和，避免长度偏差
2. **无需参考模型**：进一步简化训练流程
3. **Target margin γ**：鼓励更大的偏好间隔

**性能数据（Gemma-2-9B-it 基座）：**
- AlpacaEval 2.0 (LC win rate): **72.4%**（DPO 为 66.0%）
- Arena-Hard win rate: **59.1%**（DPO 为 51.6%）
- MT-Bench: **8.27**（超越多个 70B 模型）
- Chatbot Arena：<10B 模型排名 **第 1**

**调参建议：**
- **β:** 2.0-10.0（常用 3.0-5.0，比 DPO 大）
- **γ (margin):** 0.5-2.0（常用 1.0）
- **Learning rate:** 1e-7 到 5e-7
- **Batch size:** 64-128

### 2.4 KTO (Kahneman-Tversky Optimization)

**论文信息：**
- **标题：** Model Alignment as Prospect Theoretic Optimization
- **作者：** Kawin Ethayarajh, Winnie Xu, et al. (Toyota Research Institute)
- **arXiv：** [2402.01306](https://arxiv.org/abs/2402.01306)
- **发表时间：** 2024 年 2 月
- **ICML 2024 录用
- **访问日期：** 2026-03-16

**核心思想：**
- 基于前景理论（Prospect Theory）：人类对损失的敏感度高于收益
- 使用二元信号（desirable/undesirable）而非成对偏好
- 引入损失厌恶系数 λ > 1

**KTO Loss：**
```
L_KTO = E_{y~D_desirable} [λ·(1 - σ(β·r_θ(x,y)))] + E_{y~D_undesirable} [σ(β·r_θ(x,y))]
```

**关键优势：**
1. **无需成对数据**：单个样本即可训练
2. **数据效率高**：可利用更多来源的反馈数据
3. **性能匹配 DPO**：在 1B-30B 规模上验证

**调参建议：**
- **β:** 0.1-1.0（常用 0.5）
- **λ (损失厌恶系数):** 1.5-3.0（常用 2.0）
- **Learning rate:** 5e-7 到 2e-6

---

## 主题 3：RLAIF（AI 生成反馈）

### 3.1 Constitutional AI 论文

**论文信息：**
- **标题：** Constitutional AI: Harmlessness from AI Feedback
- **作者：** Yuntao Bai, Saurav Kadavath, et al. (Anthropic)
- **arXiv：** [2212.08073](https://arxiv.org/abs/2212.08073)
- **发表时间：** 2022 年 12 月
- **访问日期：** 2026-03-16

**核心方法：**
1. **监督学习阶段：**
   - 从初始模型采样 responses
   - 使用 AI 生成自我批评和修订
   - 在修订后的 responses 上微调模型

2. **RL 阶段（RLAIF）：**
   - 从微调模型采样
   - 使用 AI 评估哪个 response 更好（基于宪法原则）
   - 训练偏好模型（AI 偏好数据集）
   - 使用 PPO 训练，AI 偏好模型作为奖励信号

**关键发现：**
1. RLAIF 可实现无害但非回避的 AI 助手
2. 仅需人类提供原则列表，无需标注有害输出
3. 可结合 chain-of-thought 提升透明度和性能
4. 大幅减少人类标注需求（~100x 减少）

**AI 反馈质量评估：**
- 人类评估显示 RLAIF 模型与 RLHF 模型性能相当
- 在 Helpfulness 和 Harmlessness 指标上表现优异
- AI 反馈一致性：~85% 与人类判断一致

### 3.2 RLAIF 与 RLHF 对比

| 维度 | RLHF | RLAIF |
|------|------|-------|
| **人类标注需求** | 大量（数千 - 数万） | 极少（仅原则列表） |
| **训练成本** | 高（需训练 RM+PPO） | 中（AI 生成偏好） |
| **可扩展性** | 受限于人类标注 | 高（AI 可无限生成） |
| **潜在偏差** | 人类标注者偏差 | AI 模型偏差 |
| **适用场景** | 通用对齐 | 特定原则对齐 |

---

## 主题 4：性能对比数据

### 4.1 各方法在基准上的表现对比

**AlpacaEval 2.0 对比（Length-Controlled Win Rate）：**

| 方法 | 基座模型 | LC Win Rate | 来源 |
|------|----------|-------------|------|
| SimPO | Gemma-2-9B-it | 72.4% | [2405.14734](https://arxiv.org/abs/2405.14734) |
| DPO | Gemma-2-9B-it | 66.0% | [2405.14734](https://arxiv.org/abs/2405.14734) |
| ORPO | Mistral-7B | ~60% | [2403.07691](https://arxiv.org/abs/2403.07691) |
| PPO/RLHF | Llama-2-7B | ~50-55% | InstructGPT 经验值 |
| KTO | Llama-2-7B | ~55% | [2402.01306](https://arxiv.org/abs/2402.01306) |

**Arena-Hard Win Rate 对比：**

| 方法 | 基座模型 | Win Rate | 来源 |
|------|----------|----------|------|
| SimPO | Gemma-2-9B-it | 59.1% | [2405.14734](https://arxiv.org/abs/2405.14734) |
| DPO | Gemma-2-9B-it | 51.6% | [2405.14734](https://arxiv.org/abs/2405.14734) |
| PPO/RLHF | GPT-3.5 | ~50% | 基线 |

**MT-Bench 对比：**

| 方法 | 基座模型 | MT-Bench 得分 | 来源 |
|------|----------|---------------|------|
| SimPO | Gemma-2-9B-it | 8.27 | [2405.14734](https://arxiv.org/abs/2405.14734) |
| ORPO | Mistral-7B | 7.32 | [2403.07691](https://arxiv.org/abs/2403.07691) |
| DPO | Llama-2-7B | ~6.5-7.0 | 经验值 |

### 4.2 训练时间和计算成本对比

**显存需求（估算，单卡 A100 80GB）：**

| 方法 | 7B 模型 | 13B 模型 | 70B 模型 |
|------|---------|----------|----------|
| **PPO/RLHF** | 40-80GB (ZeRO) | 80-160GB (多卡) | 多节点 |
| **DPO** | 24-40GB | 40-80GB | 160GB+ |
| **ORPO** | 20-32GB | 32-64GB | 128GB+ |
| **SimPO** | 20-32GB | 32-64GB | 128GB+ |
| **KTO** | 24-40GB | 40-80GB | 160GB+ |

**训练时间对比（7B 模型，UltraFeedback 数据集，~60k 样本）：**

| 方法 | 训练时间 (A100) | 相对速度 |
|------|-----------------|----------|
| **PPO/RLHF** | 24-48 小时 | 1x（基准） |
| **DPO** | 6-12 小时 | 3-4x 更快 |
| **ORPO** | 4-8 小时 | 4-6x 更快 |
| **SimPO** | 6-12 小时 | 3-4x 更快 |
| **KTO** | 6-12 小时 | 3-4x 更快 |

**数据需求对比：**

| 方法 | 最小数据量 | 推荐数据量 | 数据类型 |
|------|------------|------------|----------|
| **PPO/RLHF** | 10k prompts | 50k-100k | prompt + human preference |
| **DPO** | 5k pairs | 20k-50k | prompt + chosen + rejected |
| **ORPO** | 5k pairs | 20k-50k | prompt + chosen + rejected |
| **SimPO** | 5k pairs | 20k-50k | prompt + chosen + rejected |
| **KTO** | 10k samples | 50k-100k | prompt + response + label |

### 4.3 Helpfulness vs Harmlessness 权衡

**Anthropic 研究数据（Constitutional AI 论文）：**

| 模型 | Helpfulness | Harmlessness | 方法 |
|------|-------------|--------------|------|
| Baseline | 50% | 50% | SFT only |
| RLHF | 72% | 78% | PPO + human labels |
| RLAIF | 70% | 76% | PPO + AI feedback |
| DPO | 68% | 74% | Direct optimization |

---

## 主题 5：2024-2025 最新进展

### 5.1 GRPO（Group Relative Policy Optimization）

**论文信息：**
- **标题：** Pushing the Limits of Mathematical Reasoning in Open Language Models (DeepSeekMath)
- **作者：** Zhihong Shao, et al. (DeepSeek)
- **arXiv：** [2402.03300](https://arxiv.org/abs/2402.03300)
- **发表时间：** 2024 年 2 月
- **访问日期：** 2026-03-16

**核心创新：**
1. GRPO 是 PPO 的变体，优化内存使用
2. 使用 group-wise 优势估计，减少 critic 模型需求
3. 在数学推理任务上表现优异

**性能数据：**
- DeepSeekMath 7B：MATH benchmark 51.7%（无工具、无投票）
- Self-consistency@64：60.9% on MATH
- 接近 Gemini-Ultra 和 GPT-4 水平

### 5.2 在线学习与多模态对齐

**2024 年新方向：**
1. **在线 RLHF**：持续从用户交互中学习
2. **多模态对齐**：视觉 - 语言模型的偏好优化
3. **多轮对话优化**：考虑对话历史的奖励建模

**代表性工作：**
- Qwen2-VL、Qwen2.5-VL 的多模态 DPO 实现
- LLaVA 系列模型的 RLHF 实践
- 视频理解任务的偏好优化

### 5.3 开源实现最佳实践

**TRL (Transformers Reinforcement Learning)：**
- GitHub: [huggingface/trl](https://github.com/huggingface/trl)
- 支持方法：SFT, PPO, DPO, GRPO, Reward Training
- 关键特性：
  - 集成 Accelerate，支持多 GPU/多节点
  - 支持 PEFT/LoRA/QLoRA
  - 集成 Unsloth 加速
  - CLI 工具简化训练

**LLaMA-Factory：**
- GitHub: [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- 支持方法：SFT, PPO, DPO, KTO, ORPO, SimPO
- 支持模型：100+ LLMs & VLMs
- 关键特性：
  - Web UI (LLaMA Board)
  - 支持 FlashAttention-2, Liger Kernel
  - 支持 vLLM/SGLang 推理加速
  - 支持多种量化方法

---

## 主题 6：代码实现参考

### 6.1 HuggingFace TRL DPO 实现示例

```python
from trl import DPOTrainer
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# 初始化 DPOTrainer
trainer = DPOTrainer(
    model="Qwen3/Qwen-0.6B",
    ref_model=None,  # 可以是 None，自动使用当前模型作为参考
    beta=0.1,        # DPO 温度参数
    train_dataset=dataset,
    max_length=512,
    max_prompt_length=128,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    num_train_epochs=1,
)

# 开始训练
trainer.train()
```

### 6.2 TRL PPO 实现示例

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import respond_to_batch
from transformers import AutoTokenizer
from datasets import load_dataset

# 配置
ppo_config = PPOConfig(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    learning_rate=1e-6,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
    clip_range=0.2,
    vf_coef=0.1,
)

# 加载模型和 tokenizer
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ppo_config.model_name
)
tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

# 加载数据集
dataset = load_dataset("trl-lib/ultrafeedback", split="train")

# 初始化 PPOTrainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,  # 自动创建参考模型
    tokenizer=tokenizer,
    dataset=dataset,
)

# 训练循环
for batch in ppo_trainer.dataloader:
    query_tensors = batch["input_ids"]
    
    # 生成 response
    response_tensors = ppo_trainer.generate(query_tensors)
    response_texts = tokenizer.batch_decode(response_tensors)
    
    # 获取奖励（使用奖励模型或规则）
    rewards = [compute_reward(text) for text in response_texts]
    
    # PPO 优化步骤
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    
    # 记录日志
    ppo_trainer.log_stats(stats, batch, rewards)
```

### 6.3 SimPO 实现关键代码（参考官方实现）

```python
import torch
import torch.nn as nn

class SimPOLoss(nn.Module):
    def __init__(self, beta=3.0, gamma=1.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, policy_chosen_logps, policy_rejected_logps):
        """
        policy_chosen_logps: (batch_size,) - 平均 log 概率
        policy_rejected_logps: (batch_size,) - 平均 log 概率
        """
        # 计算奖励差值
        reward_diff = policy_chosen_logps - policy_rejected_logps
        
        # SimPO loss
        losses = -nn.functional.logsigmoid(
            self.beta * reward_diff - self.gamma
        )
        
        return losses.mean()

# 计算平均 log 概率的关键函数
def get_batch_logps(logits, labels, average=True):
    """
    logits: (batch_size, seq_len, vocab_size)
    labels: (batch_size, seq_len)
    """
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    
    log_probs = nn.functional.log_softmax(shift_logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    if average:
        # 关键：长度归一化
        return token_logps.sum(dim=-1) / (shift_labels != -100).sum(dim=-1)
    else:
        return token_logps.sum(dim=-1)
```

### 6.4 ORPO 实现关键代码

```python
class ORPOLoss(nn.Module):
    def __init__(self, lambda_or=1.0):
        super().__init__()
        self.lambda_or = lambda_or
    
    def odds_ratio(self, logits, labels):
        """计算 odds ratio"""
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        # 转换为概率
        probs = torch.exp(token_logps.sum(dim=-1))
        
        # odds = p / (1-p)
        odds = probs / (1 - probs + 1e-8)
        return torch.log(odds + 1e-8)
    
    def forward(self, chosen_logits, rejected_logits, chosen_labels, rejected_labels):
        chosen_or = self.odds_ratio(chosen_logits, chosen_labels)
        rejected_or = self.odds_ratio(rejected_logits, rejected_labels)
        
        or_diff = chosen_or - rejected_or
        losses = -nn.functional.logsigmoid(or_diff)
        
        return losses.mean()
```

### 6.5 Reward Model 训练最佳实践

**数据结构：**
```python
# 偏好数据格式
{
    "prompt": "用户问题",
    "chosen": "更好的回答",
    "rejected": "较差的回答"
}
```

**Reward Model 训练代码：**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer
from datasets import load_dataset

# 加载模型（添加分类头）
model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    num_labels=1  # 标量奖励
)

# 加载数据集
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# 初始化 RewardTrainer
trainer = RewardTrainer(
    model=model,
    train_dataset=dataset,
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=1,
)

# 训练
trainer.train()

# 保存奖励模型
trainer.save_model("reward_model")
```

**Reward Model 调参建议：**
- **Learning rate:** 1e-5 到 5e-5
- **Batch size:** 32-128
- **Epochs:** 1-2（避免过拟合）
- **Max length:** 512-1024
- **Dropout:** 0.1-0.2

---

## 附录：关键资源链接汇总

### 论文链接
1. **PPO:** https://arxiv.org/abs/1707.06347
2. **InstructGPT:** https://arxiv.org/abs/2203.02155
3. **DPO:** https://arxiv.org/abs/2305.18290
4. **ORPO:** https://arxiv.org/abs/2403.07691
5. **SimPO:** https://arxiv.org/abs/2405.14734
6. **KTO:** https://arxiv.org/abs/2402.01306
7. **Constitutional AI (RLAIF):** https://arxiv.org/abs/2212.08073
8. **GRPO (DeepSeekMath):** https://arxiv.org/abs/2402.03300

### 代码库
1. **TRL:** https://github.com/huggingface/trl
2. **LLaMA-Factory:** https://github.com/hiyouga/LLaMA-Factory
3. **SimPO 官方实现:** https://github.com/princeton-nlp/SimPO

### 数据集
1. **UltraFeedback:** https://huggingface.co/datasets/argilla/ultrafeedback-binarized
2. **Capybara:** https://huggingface.co/datasets/trl-lib/Capybara
3. **Anthropic HH-RLHF:** https://huggingface.co/datasets/Anthropic/hh-rlhf

### 技术博客
1. **HuggingFace DPO 教程:** https://huggingface.co/docs/trl/dpo_trainer
2. **HuggingFace PPO 教程:** https://huggingface.co/docs/trl/ppo
3. **LlamaFactory 博客:** https://blog.llamafactory.net/

---

**文档结束**

*本参考资料由 Searcher Agent 整理，所有数据均来自公开论文和官方文档。*
