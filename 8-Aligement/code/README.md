# 大模型强化学习代码项目

本项目实现了大语言模型强化学习的主流算法，包括 PPO、DPO、ORPO 和 SimPO。

## 项目结构

```
code/
├── README.md              # 项目说明
├── requirements.txt       # Python 依赖
├── config.yaml           # 训练配置
├── src/
│   ├── data/             # 数据预处理
│   ├── models/           # 模型定义
│   ├── algorithms/       # 核心算法实现
│   └── train/           # 训练脚本
├── scripts/             # Shell 运行脚本
└── tests/               # 测试代码
```

## 环境安装

### 系统要求
- Python 3.9+
- GPU (推荐 NVIDIA，16GB+ 显存)
- CUDA 11.8+

### 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备

准备偏好数据集，格式为 JSONL：
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

### 2. 配置训练

编辑 `config.yaml` 文件，设置：
- 模型路径
- 训练超参数
- 数据路径

### 3. 运行训练

#### SFT (监督微调)
```bash
bash scripts/run_sft.sh
```

#### DPO (直接偏好优化)
```bash
bash scripts/run_dpo.sh
```

#### PPO (近端策略优化)
```bash
bash scripts/run_ppo.sh
```

## 算法说明

### PPO (Proximal Policy Optimization)
- 基于 Actor-Critic 架构
- 使用 GAE 计算优势函数
- Clip 机制保证训练稳定性
- 推荐配置见 `config.yaml` 的 `ppo` 部分

### DPO (Direct Preference Optimization)
- 无需显式奖励模型
- 直接优化偏好损失
- 需要参考模型
- 推荐配置见 `config.yaml` 的 `dpo` 部分

### ORPO (Odds Ratio Policy Optimization)
- 无需参考模型
- 结合 SFT 和 Odds Ratio 损失
- 计算效率高
- 推荐配置见 `config.yaml` 的 `orpo` 部分

### SimPO (Simple Preference Optimization)
- 长度归一化的平均 log 概率
- 无需参考模型
- 使用 target margin γ
- 推荐配置见 `config.yaml` 的 `simpo` 部分

## 配置文件说明

`config.yaml` 包含所有超参数：

```yaml
# 通用配置
model:
  base_model: "meta-llama/Llama-2-7b-hf"
  max_length: 512

# PPO 配置
ppo:
  clip_epsilon: 0.2
  value_coeff: 0.5
  entropy_coeff: 0.01
  gae_lambda: 0.95
  gamma: 0.99

# DPO 配置
dpo:
  beta: 0.1
  label_smoothing: 0.0

# ORPO 配置
orpo:
  lambda: 0.5
  beta: 0.1

# SimPO 配置
simpo:
  gamma: 0.5
  beta: 0.1
```

## 测试

运行损失函数测试：
```bash
python -m pytest tests/test_losses.py -v
```

## 核心 API

### 数据预处理
```python
from src.data.preprocessing import PreferenceDataset, collate_fn

dataset = PreferenceDataset(data_path="data.jsonl")
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
```

### PPO 算法
```python
from src.algorithms.ppo import PPOTrainer

trainer = PPOTrainer(config=config)
loss = trainer.compute_loss(policy_outputs, value_outputs, advantages)
```

### DPO 算法
```python
from src.algorithms.dpo import DPOTrainer

trainer = DPOTrainer(config=config, ref_model=ref_model)
loss = trainer.compute_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
```

## 调参建议

### PPO
- `clip_epsilon`: 0.1-0.3 (默认 0.2)
- `value_coeff`: 0.1-0.5 (默认 0.5)
- `entropy_coeff`: 0.001-0.1 (默认 0.01)
- 学习率：1e-6 到 1e-4

### DPO
- `beta`: 0.1-0.5 (默认 0.1)
- 学习率：5e-7 到 5e-6
- batch size: 16-64

### ORPO
- `lambda`: 0.3-0.7 (默认 0.5)
- `beta`: 0.05-0.2 (默认 0.1)

### SimPO
- `gamma`: 0.3-0.7 (默认 0.5)
- `beta`: 0.05-0.2 (默认 0.1)

## 常见问题

### Q: 训练不稳定怎么办？
A: 尝试降低学习率，增加 `clip_epsilon`，或减小 batch size。

### Q: 显存不足怎么办？
A: 使用 gradient accumulation 或减小 batch size。

### Q: 如何选择合适的 beta 值？
A: 从 0.1 开始，根据验证集效果调整。

## 许可证

MIT License

## 参考文献

1. PPO: Proximal Policy Optimization Algorithms (Schulman et al., 2017)
2. DPO: Direct Preference Optimization (Rafailov et al., 2023)
3. ORPO: Monotonic Policy Optimization (Hong et al., 2024)
4. SimPO: Simple Preference Optimization (Meng et al., 2024)
