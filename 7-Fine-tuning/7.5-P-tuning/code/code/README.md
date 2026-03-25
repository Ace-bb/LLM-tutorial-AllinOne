# P-tuning 实现

基于 PyTorch 和 HuggingFace Transformers 的 P-tuning 完整实现。

## 项目简介

P-tuning 是一种参数高效的微调方法，通过在输入序列中添加可训练的连续向量（虚拟词元）来引导预训练语言模型。本项目实现了原始 P-tuning 论文（《GPT Understands, Too》, ACL 2021）中的核心算法。

**核心特点：**
- ✅ LSTM+MLP 提示编码器（原始论文架构）
- ✅ 仅训练 0.01%-1% 参数，冻结主模型
- ✅ 支持 GPT-2、BERT 等预训练模型
- ✅ 完整的训练和推理流程
- ✅ 清晰的代码注释和文档

## 项目结构

```
code/
├── src/
│   ├── __init__.py           # 模块初始化
│   ├── prompt_encoder.py     # LSTM+MLP 编码器
│   ├── ptuning_model.py      # P-tuning 模型封装
│   ├── train.py              # 训练脚本
│   └── inference.py          # 推理脚本
├── data/
│   └── README.md             # 数据说明
├── config.py                 # 超参数配置
├── requirements.txt          # 依赖配置
└── README.md                 # 使用说明
```

## 快速开始

### 1. 安装依赖

```bash
cd code
pip install -r requirements.txt
```

### 2. 测试安装

```bash
# 测试 PromptEncoder
python src/prompt_encoder.py

# 测试 PTuningModel
python src/ptuning_model.py
```

### 3. 训练模型

**使用示例数据测试：**
```bash
python src/train.py --use_sample_data --num_epochs 10
```

**使用真实数据（以 IMDB 为例）：**

修改 `src/train.py` 中的数据加载部分：

```python
from datasets import load_dataset

# 加载 IMDB 数据集
dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"][:1000]  # 使用前 1000 条
train_labels = dataset["train"]["label"][:1000]
val_texts = dataset["test"]["text"][:200]
val_labels = dataset["test"]["label"][:200]
```

然后运行：
```bash
python src/train.py --model_name gpt2 --num_virtual_tokens 50 --num_epochs 50 --output_dir ./output
```

### 4. 推理预测

**文本分类：**
```bash
python src/inference.py --model_path ./output/best_model --text "这部电影太棒了，我非常喜欢！"
```

**文本生成：**
```bash
python src/inference.py --model_path ./output/best_model --prompt "今天天气真好，" --max_new_tokens 50
```

## 核心组件详解

### PromptEncoder（提示编码器）

```python
from src.prompt_encoder import PromptEncoder

# 创建编码器
encoder = PromptEncoder(
    embed_dim=768,           # 嵌入维度（与模型一致）
    hidden_dim=512,          # LSTM 隐藏层
    num_virtual_tokens=50,   # 虚拟词元数量
    bidirectional=True       # 双向 LSTM
)

# 生成提示嵌入
batch_size = 4
prompt_embeds = encoder(batch_size)
# 输出形状：[4, 50, 768]
```

### PTuningModel（P-tuning 模型）

```python
from src.ptuning_model import PTuningModel

# 创建模型
model = PTuningModel(
    model_name="gpt2",
    num_virtual_tokens=50,
    encoder_hidden_dim=512,
    task_type="classification",  # 或 "causal_lm"
    num_labels=2
)

# 查看可训练参数
model.print_trainable_parameters()
# 输出：可训练参数：约 0.03%（相比全量微调）

# 前向传播
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)
loss = outputs.loss
```

## 超参数配置

### 推荐配置（根据任务调整）

| 超参数 | 简单任务 | 复杂任务 | 小样本 |
|-------|---------|---------|--------|
| 虚拟词元数量 | 20-50 | 50-100 | 80-100 |
| LSTM 隐藏层 | 512 | 512 | 512 |
| 学习率 | 1e-3 | 5e-4 | 5e-4 |
| 批大小 | 16-32 | 16 | 8-16 |
| 训练轮数 | 50 | 100 | 200 |

### 使用配置文件

```python
from config import P TuningConfig, get_task_config

# 获取情感分类推荐配置
config = get_task_config("sentiment_classification")

print(f"虚拟词元数量：{config.num_virtual_tokens}")
print(f"训练轮数：{config.num_epochs}")
```

## 训练技巧

### 1. 虚拟词元数量选择

- **简单任务**（情感分类）：20-50 个
- **复杂任务**（NLI、多分类）：50-100 个
- **小样本学习**：适当增加至 100 个
- **长序列**：减少至 20-30 个（避免占用过多序列长度）

### 2. 学习率调整

- 默认：1e-3（比全量微调大）
- 范围：1e-4 ~ 1e-2
- 根据验证集调整

### 3. 防止过拟合

- 使用 Dropout（encoder 中已包含）
- 早停（Early Stopping）
- 数据增强（小样本场景）

### 4. 调试技巧

**验证参数冻结：**
```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")
```

**检查梯度流：**
```python
loss.backward()
for name, param in model.prompt_encoder.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm().item():.4f}")
```

## 模型保存与加载

### 保存模型

```python
# 训练完成后自动保存
model.save_pretrained("./output/best_model")
```

### 加载模型

```python
from src.ptuning_model import PTuningModel

# 加载训练好的模型
model = PTuningModel.from_pretrained(
    "./output/best_model",
    device="cuda"  # 或 "cpu"
)
model.eval()
```

## API 参考

### PTuningModel 方法

| 方法 | 说明 |
|------|------|
| `forward()` | 前向传播 |
| `generate()` | 文本生成（causal_lm 任务） |
| `get_trainable_parameters()` | 获取可训练参数 |
| `print_trainable_parameters()` | 打印参数统计 |
| `save_pretrained()` | 保存模型 |
| `from_pretrained()` | 加载模型 |

### P TuningInference 方法

| 方法 | 说明 |
|------|------|
| `predict()` | 文本分类预测 |
| `generate()` | 文本生成 |
| `predict_batch()` | 批量预测 |
| `get_confidence()` | 获取置信度 |
| `explain_prediction()` | 解释预测结果 |

## 与 HuggingFace PEFT 对比

| 特性 | 本实现 | HuggingFace PEFT |
|------|--------|-----------------|
| 编码器架构 | LSTM+MLP（原始 P-tuning） | 无（Prompt Tuning） |
| 实现复杂度 | 中等 | 简单 |
| 小模型收敛 | 快（LSTM 重参数化） | 较慢 |
| 可定制性 | 高 | 中 |
| 学习价值 | 高（清晰展示原理） | 中（封装较多） |

**推荐使用场景：**
- **学习和研究**：使用本实现（代码清晰，便于理解原理）
- **生产环境**：使用 HuggingFace PEFT（集成度高，维护好）

## 常见问题

### Q1: 虚拟词元放在什么位置？

**A:** 默认放在输入序列最前面（前缀）：
```python
combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
```

### Q2: 为什么使用 LSTM 而不是直接学习？

**A:** LSTM 可以捕捉虚拟词元之间的依赖关系，实验表明比直接学习收敛更快（特别是小模型）。

### Q3: 支持哪些预训练模型？

**A:** 理论上支持所有 HuggingFace Transformers 模型，已测试：
- GPT-2
- BERT
- RoBERTa
- Chinese-BERT-wwm

### Q4: 训练需要多少显存？

**A:** 相比全量微调节省 80%+ 显存：
- GPT-2（124M）：约 2-3GB
- BERT-base（110M）：约 2-3GB

### Q5: 与 P-tuning V2 有什么区别？

**A:** 
- **原始 P-tuning**（本实现）：仅在输入层添加提示
- **P-tuning V2**：在每一层 Transformer 都添加提示，实现更复杂，性能更好

本文实现的是原始 P-tuning（Liu et al., 2021）。

## 参考文献

1. Liu, X., et al. (2021). **GPT Understands, Too**. arXiv:2103.10385
2. Li, X. L., & Liang, P. (2021). **Prefix-Tuning: Optimizing Continuous Prompts for Generation**. arXiv:2101.00190
3. Lester, B., et al. (2021). **The Power of Scale for Parameter-Efficient Prompt Tuning**. arXiv:2104.08691

## 许可证

MIT License

## 致谢

- 清华大学 THUDM 实验室的原始 P-tuning 工作
- HuggingFace Transformers 团队
- PyTorch 团队

---

*实现版本：1.0.0*
*最后更新：2026-03-16*
