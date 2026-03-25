# Prefix-tuning 完整实现

本项目提供了 Prefix-tuning 的完整 PyTorch 实现，支持 GPT-2 和 BART 模型。

## 📋 目录

- [简介](#简介)
- [安装](#安装)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [核心 API](#核心-api)
- [训练指南](#训练指南)
- [推理指南](#推理指南)
- [配置说明](#配置说明)

## 简介

Prefix-tuning 是一种参数高效微调（PEFT）方法，其核心思想是：

- **冻结**预训练模型的全部参数
- **添加**可训练的连续前缀向量到每一层 Transformer
- **仅训练**前缀参数（通常仅为总参数的 0.1%-1%）

相比全量微调，Prefix-tuning 具有：
- ✅ 极低的显存占用
- ✅ 更快的训练速度
- ✅ 避免灾难性遗忘
- ✅ 便于多任务部署

## 安装

```bash
# 克隆或下载本项目
cd prefix-tuning-implementation

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; import transformers; print('安装成功！')"
```

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- GPU（推荐，可选 CPU）

## 快速开始

### 1. 训练模型

```bash
# 训练 GPT-2 进行文本生成
python scripts/train.py \
    --model_name gpt2 \
    --model_type causal \
    --prefix_length 20 \
    --output_dir ./output/gpt2-prefix \
    --num_train_epochs 10

# 训练 BART 进行序列到序列任务
python scripts/train.py \
    --model_name facebook/bart-base \
    --model_type seq2seq \
    --prefix_length 30 \
    --output_dir ./output/bart-prefix \
    --num_train_epochs 15
```

### 2. 推理生成

```bash
# 使用训练好的模型进行文本生成
python scripts/inference.py \
    --model_path ./output/gpt2-prefix \
    --prompt "人工智能是" \
    --max_length 100

# 文本摘要任务
python scripts/inference.py \
    --model_path ./output/bart-prefix \
    --prompt "这是一段需要摘要的长文本..." \
    --task summarization \
    --max_length 50
```

## 项目结构

```
prefix-tuning-implementation/
├── src/
│   ├── __init__.py              # 包初始化
│   ├── prefix_model.py          # Prefix-tuning 核心实现
│   ├── trainer.py               # 训练逻辑
│   └── utils.py                 # 工具函数
├── configs/
│   └── config.yaml              # 超参数配置示例
├── scripts/
│   ├── train.py                 # 训练脚本
│   └── inference.py             # 推理脚本
├── requirements.txt             # 依赖列表
└── README.md                    # 本文档
```

## 核心 API

### PrefixTuningConfig

配置类，用于设置 Prefix-tuning 参数：

```python
from src.prefix_model import PrefixTuningConfig

config = PrefixTuningConfig(
    model_name_or_path="gpt2",      # 预训练模型
    model_type="causal",            # 'causal' 或 'seq2seq'
    prefix_length=20,               # 前缀长度
    bottleneck_size=512,            # 重参数化瓶颈层
    dropout=0.1,                    # Dropout 率
    prefix_projection=True,         # 使用重参数化
)
```

### PrefixTuningModel

主模型类，包装预训练模型并添加前缀：

```python
from src.prefix_model import PrefixTuningModel

model = PrefixTuningModel(config)
model.to("cuda")

# 前向传播
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels  # 训练时提供
)

# 生成文本
generated = model.generate(
    input_ids=input_ids,
    max_length=100,
    temperature=0.7
)
```

### PrefixTrainer

训练器类，提供完整的训练循环：

```python
from src.trainer import PrefixTrainer, TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=10,
    learning_rate=5e-4,
    per_device_train_batch_size=4,
)

trainer = PrefixTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=args
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model()
```

## 训练指南

### 数据集准备

本项目支持 HuggingFace `datasets` 库的所有数据集。常用数据集：

| 任务类型 | 数据集 | 说明 |
|---------|--------|------|
| 文本生成 | `wikitext-2` | Wikipedia 文本 |
| 问答 | `squad` | SQuAD 问答数据集 |
| 摘要 | `cnn_dailymail` | CNN 新闻摘要 |
| 翻译 | `wmt16` | WMT 翻译数据集 |

自定义数据集示例：

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten().clone()
        }
```

### 超参数调优

关键超参数推荐范围：

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `prefix_length` | 10-30 | 任务越复杂，值越大 |
| `learning_rate` | 1e-3 - 5e-2 | 通常高于全量微调 |
| `batch_size` | 4-16 | 根据显存调整 |
| `bottleneck_size` | 256-1024 | 影响表达能力 |

### 训练技巧

1. **使用梯度累积**：显存不足时增加 `gradient_accumulation_steps`
2. **混合精度训练**：启用 `--fp16` 加速训练
3. **早停策略**：监控验证集损失，防止过拟合
4. **学习率调度**：使用 warmup + cosine decay

## 推理指南

### 文本生成

```python
python scripts/inference.py \
    --model_path ./output/gpt2-prefix \
    --prompt "从前有个" \
    --max_length 200 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.95
```

### 文本摘要

```python
python scripts/inference.py \
    --model_path ./output/bart-prefix \
    --prompt "长文本内容..." \
    --task summarization \
    --max_length 100 \
    --num_beams 5
```

### 批量推理

```python
from src.prefix_model import PrefixTuningModel
from transformers import AutoTokenizer
import torch

model = PrefixTuningModel.from_pretrained("./output/gpt2-prefix")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompts = ["提示 1", "提示 2", "提示 3"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100)

results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

## 配置说明

完整配置参数见 `configs/config.yaml`：

```yaml
# 模型配置
model_name_or_path: "gpt2"
model_type: "causal"

# 前缀配置
prefix_length: 20
bottleneck_size: 512
prefix_projection: true

# 训练配置
learning_rate: 5e-4
num_train_epochs: 10
per_device_train_batch_size: 4

# 输出配置
output_dir: "./output"
save_steps: 500
logging_steps: 10
```

## 常见问题

### Q: 训练不收敛怎么办？

- 检查学习率是否过高（尝试降低到 1e-4）
- 增加前缀长度
- 确保模型参数正确冻结
- 使用重参数化技巧（`prefix_projection=True`）

### Q: 显存不足如何优化？

- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 启用 `--fp16` 混合精度
- 减少 `prefix_length`

### Q: 如何保存和加载模型？

```python
# 保存
trainer.save_model("./my-prefix-model")

# 加载
model = PrefixTuningModel.from_pretrained("./my-prefix-model", config)
```

## 参考资料

- 原始论文：[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- HuggingFace PEFT 库：https://github.com/huggingface/peft
- 官方实现：https://github.com/XiangLi1999/PrefixTuning

## 许可证

MIT License

---

*本实现用于教学和研究目的，欢迎贡献和改进！*
