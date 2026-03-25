# 数据说明

本目录用于存放训练和测试数据。

## 推荐数据集

### 1. 情感分类任务

**IMDB 电影评论数据集**
- 来源：https://ai.stanford.edu/~amaas/data/sentiment/
- 规模：50,000 条评论（25,000 训练 + 25,000 测试）
- 标签：正面/负面（二分类）
- 使用 HuggingFace 加载：
  ```python
  from datasets import load_dataset
  dataset = load_dataset("imdb")
  ```

**SST-2 情感分析**
- 来源：https://gluebenchmark.com/
- 规模：约 70,000 条
- 标签：正面/负面
- 使用 HuggingFace 加载：
  ```python
  from datasets import load_dataset
  dataset = load_dataset("glue", "sst2")
  ```

### 2. 自然语言推理（NLI）

**MNLI**
- 来源：https://gluebenchmark.com/
- 标签：蕴含/矛盾/中性（三分类）
- 使用 HuggingFace 加载：
  ```python
  dataset = load_dataset("glue", "mnli")
  ```

**RTE**
- 来源：https://gluebenchmark.com/
- 标签：蕴含/非蕴含（二分类）
- 使用 HuggingFace 加载：
  ```python
  dataset = load_dataset("glue", "rte")
  ```

### 3. 自定义数据

如果需要使用自定义数据，请按照以下格式组织：

**训练数据格式（JSON）:**
```json
[
    {"text": "文本内容 1", "label": 0},
    {"text": "文本内容 2", "label": 1},
    ...
]
```

**标签映射（label_map.json）:**
```json
{
    "0": "负面",
    "1": "正面"
}
```

## 数据预处理

训练脚本会自动处理：
- 分词（使用模型对应的 tokenizer）
- Padding 到统一长度
- Truncation（超过最大长度的截断）
- 添加特殊 token

## 数据增强（可选）

对于小样本场景，可以考虑：
- 同义词替换
- 回译（Back Translation）
- EDA (Easy Data Augmentation)

## 注意事项

1. 确保训练集和测试集分布一致
2. 处理类别不平衡问题（如需要）
3. 对于中文数据，使用支持中文的 tokenizer
4. 数据量建议：至少 1000+ 样本用于训练
