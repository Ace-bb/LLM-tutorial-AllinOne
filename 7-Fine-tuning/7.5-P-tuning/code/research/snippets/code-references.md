# P-tuning 代码实现参考

> 包含官方仓库、HuggingFace 实现、关键代码片段

---

## 一、官方代码仓库

### 1.1 THUDM/P-tuning（原始实现）

**仓库地址：** https://github.com/THUDM/P-tuning

**论文对应：** 《GPT Understands, Too》(arxiv:2103.10385)

**目录结构：**
```
P-tuning/
├── models/
│   └── modeling_gpt2.py        # GPT-2 模型修改（添加 P-tuning）
├── data_utils/
│   └── data_loader.py          # 数据加载
├── LAMA/                        # LAMA 知识探测实验
│   ├── run_lama.py
│   └── requirements.txt
├── FewGLUE_32dev/              # 少样本 SuperGLUE 实验
│   ├── run_glue.py
│   └── requirements.txt
└── README.md
```

**核心代码片段（官方实现简化）：**

```python
# models/modeling_gpt2.py 关键修改

class GPT2LMHeadModelWithPTuning(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # 虚拟词元数量
        self.num_virtual_tokens = config.num_virtual_tokens
        
        # 虚拟词元嵌入
        self.virtual_tokens = nn.Embedding(self.num_virtual_tokens, config.n_embd)
        
        # 提示编码器（LSTM + MLP）
        self.prompt_encoder = nn.Sequential(
            nn.LSTM(config.n_embd, config.hidden_size, batch_first=True, bidirectional=True),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.n_embd)
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        
        # 获取虚拟词元嵌入
        virtual_tokens_embeds = self.virtual_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 编码器处理
        prompt_embeds = self.prompt_encoder(virtual_tokens_embeds)[0]  # LSTM 输出
        
        # 获取输入嵌入
        input_embeds = self.transformer.wte(input_ids)
        
        # 拼接
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        # 调整 attention_mask
        prompt_attention_mask = torch.ones(
            (batch_size, self.num_virtual_tokens), 
            device=input_ids.device
        )
        combined_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        
        # 通过 Transformer（冻结）
        outputs = self.transformer(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask
        )
        
        # 计算损失（仅基于实际输入部分）
        logits = self.lm_head(outputs[0][:, self.num_virtual_tokens:, :])
        
        return logits
```

**训练配置（官方默认）：**
```python
# LAMA/run_lama.py 中的配置
training_args = {
    'num_virtual_tokens': 50,
    'hidden_size': 512,
    'learning_rate': 1e-3,
    'batch_size': 16,
    'num_epochs': 100,
    'weight_decay': 0.01,
}
```

---

### 1.2 THUDM/P-tuning-v2（V2 实现）

**仓库地址：** https://github.com/THUDM/P-tuning-v2

**重要说明：** 这是 P-tuning V2 实现，**不是**本文讲解的原始 P-tuning。

**关键区别：**
- V2 在每一层 Transformer 都添加提示
- 实现更复杂
- 参数量更多（0.1%-3%）

**参考用途：** 了解技术演进，不作为本文代码参考。

---

## 二、HuggingFace PEFT 实现

### 2.1 使用 PEFT 库（推荐）

**仓库地址：** https://github.com/huggingface/peft

**安装：**
```bash
pip install peft
```

**基本用法：**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, TaskType, get_peft_model

# 加载基础模型
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 配置 Prompt Tuning
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=50,           # 虚拟词元数量
    prompt_tuning_init="TEXT",       # 或 "RANDOM"
    tokenizer_name_or_path=model_id,
    prompt_tuning_init_text="Classify if this text is positive or negative:",
)

# 获取 PEFT 模型
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# 输出：trainable params: 38,400 || all params: 124,439,808 || trainable%: 0.030858

# 训练（使用 transformers Trainer）
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    num_train_epochs=100,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

**注意：** HuggingFace PEFT 实现的是 **Prompt Tuning**（Lester et al.），没有 LSTM+MLP 编码器，是直接学习虚拟词元嵌入。

---

### 2.2 PEFT 库中的 Prompt Tuning 配置

```python
from peft import PromptTuningConfig, PromptTuningInit

# 完整配置选项
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    
    # 虚拟词元数量
    num_virtual_tokens=50,
    
    # 初始化方式
    prompt_tuning_init=PromptTuningInit.TEXT,  # 或 RANDOM
    
    # 如果使用 TEXT 初始化
    tokenizer_name_or_path="gpt2",
    prompt_tuning_init_text="Classify sentiment:",
    
    # 其他配置
    inference_mode=False,  # 训练时设为 False
)
```

**与原始 P-tuning 的区别：**
- PEFT 的 Prompt Tuning **没有** LSTM+MLP 编码器
- 直接学习虚拟词元嵌入（类似 Prompt Tuning 论文）
- 实现更简单，但收敛可能稍慢（小模型）

---

## 三、自定义实现（推荐用于学习）

### 3.1 完整实现示例

```python
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class PromptEncoder(nn.Module):
    """P-tuning 提示编码器（LSTM + MLP）"""
    
    def __init__(self, embed_dim, hidden_dim=512, num_virtual_tokens=50):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        
        # 虚拟词元嵌入（可训练）
        self.virtual_tokens = nn.Embedding(num_virtual_tokens, embed_dim)
        nn.init.normal_(self.virtual_tokens.weight, std=0.02)
        
        # LSTM 编码器（双向）
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
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
        # 获取虚拟词元嵌入 [batch_size, n_prompt, embed_dim]
        token_embeds = self.virtual_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # LSTM 编码 [batch_size, n_prompt, hidden_dim*2]
        lstm_out, _ = self.lstm(token_embeds)
        
        # MLP 映射到嵌入空间 [batch_size, n_prompt, embed_dim]
        prompt_embeds = self.mlp(lstm_out)
        
        return prompt_embeds


class PTuningModel(nn.Module):
    """封装 GPT-2 + P-tuning"""
    
    def __init__(self, model_name='gpt2', num_virtual_tokens=50):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        
        # 加载预训练模型
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        embed_dim = self.model.config.n_embd
        
        # 冻结主模型参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 提示编码器
        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            hidden_dim=512,
            num_virtual_tokens=num_virtual_tokens
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        
        # 获取提示嵌入
        prompt_embeds = self.prompt_encoder(batch_size)
        
        # 获取输入嵌入
        input_embeds = self.model.get_input_embeddings()(input_ids)
        
        # 拼接提示和输入
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        # 调整 attention_mask
        prompt_attention_mask = torch.ones(
            (batch_size, self.num_virtual_tokens), 
            device=input_ids.device
        )
        combined_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        
        # 通过模型
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=labels
        )
        
        return outputs
    
    def get_trainable_params(self):
        """获取可训练参数（仅提示编码器）"""
        return self.prompt_encoder.parameters()


# 使用示例
model = PTuningModel(model_name='gpt2', num_virtual_tokens=50)

# 优化器（仅优化提示编码器）
optimizer = torch.optim.AdamW(
    model.get_trainable_params(),
    lr=1e-3,
    weight_decay=0.01
)

# 训练循环
model.train()
for epoch in range(100):
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

### 3.2 数据加载示例

```python
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 创建数据集
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

train_dataset = PromptDataset(
    texts=train_texts,
    labels=train_labels,
    tokenizer=tokenizer
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)
```

---

### 3.3 推理示例

```python
@torch.no_grad()
def predict(model, tokenizer, text, num_virtual_tokens=50):
    model.eval()
    
    # 分词
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True
    )
    
    # 生成（需要特殊处理虚拟词元）
    # 注意：P-tuning 推理需要保持虚拟词元在输入前缀
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=50,
        num_return_sequences=1
    )
    
    # 解码（跳过虚拟词元部分）
    generated_text = tokenizer.decode(
        outputs[0][num_virtual_tokens:],  # 跳过虚拟词元
        skip_special_tokens=True
    )
    
    return generated_text

# 使用示例
result = predict(model, tokenizer, "This movie is great!", num_virtual_tokens=50)
print(f"Prediction: {result}")
```

---

## 四、关键实现细节

### 4.1 虚拟词元位置

**原始 P-tuning：** 虚拟词元放在输入序列**最前面**（前缀）

```python
# 拼接顺序：[prompt_embeds, input_embeds]
combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
```

**可选变体：** 根据任务可放在中间或后面

```python
# 放在后面（后缀）
combined_embeds = torch.cat([input_embeds, prompt_embeds], dim=1)

# 放在中间
mid_point = input_embeds.shape[1] // 2
combined_embeds = torch.cat([
    input_embeds[:, :mid_point, :],
    prompt_embeds,
    input_embeds[:, mid_point:, :]
], dim=1)
```

---

### 4.2 损失计算

**关键：** 损失仅基于实际输入部分，排除虚拟词元。

```python
# 方法 1：在模型内部处理
outputs = self.model(inputs_embeds=combined_embeds, ...)
logits = outputs.logits[:, self.num_virtual_tokens:, :]  # 跳过虚拟词元

# 方法 2：使用 labels 自动处理（推荐）
# transformers 库会自动根据 labels 计算有效 token 的损失
loss = outputs.loss  # labels 中 -100 的位置会被忽略
```

---

### 4.3 保存和加载

```python
# 保存（仅保存提示编码器）
torch.save({
    'prompt_encoder': model.prompt_encoder.state_dict(),
    'config': {
        'num_virtual_tokens': 50,
        'hidden_dim': 512,
    }
}, 'ptuning_checkpoint.pt')

# 加载
checkpoint = torch.load('ptuning_checkpoint.pt')
model.prompt_encoder.load_state_dict(checkpoint['prompt_encoder'])
```

---

## 五、调试技巧

### 5.1 验证参数冻结

```python
# 检查哪些参数可训练
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")
    else:
        print(f"Frozen: {name}")

# 预期输出：
# Trainable: prompt_encoder.virtual_tokens.weight
# Trainable: prompt_encoder.lstm.*
# Trainable: prompt_encoder.mlp.*
# Frozen: model.transformer.*
# Frozen: model.lm_head.*
```

### 5.2 检查梯度流

```python
# 训练一步后检查梯度
outputs = model(**batch)
loss = outputs.loss
loss.backward()

# 检查提示编码器是否有梯度
for name, param in model.prompt_encoder.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm().item():.4f}")
    else:
        print(f"{name}: NO GRADIENT")

# 检查主模型是否确实冻结
for name, param in model.model.named_parameters():
    if param.grad is not None and param.grad.norm().item() > 0:
        print(f"WARNING: {name} has gradient (should be frozen)")
```

---

## 六、常见问题

### Q1: 虚拟词元数量如何选择？

**A:** 
- 简单任务（情感分类）：20-50
- 复杂任务（NLI、多分类）：50-100
- 小样本学习：适当增加
- 长序列：适当减少

### Q2: 为什么使用 LSTM 而不是直接学习？

**A:**
- LSTM 捕捉虚拟词元之间的依赖关系
- 重参数化加速收敛（特别是小模型）
- 实验表明比直接学习更快收敛

### Q3: 学习率设置多少合适？

**A:**
- 推荐 1e-3（比全量微调大）
- 因为仅训练少量参数，可以承受更大学习率
- 范围：1e-4 ~ 1e-2，根据验证集调整

### Q4: 如何处理不同长度的输入？

**A:**
- 虚拟词元数量固定
- 输入序列 padding 到统一长度
- attention_mask 正确处理 padding 位置

---

*代码片段整理时间：2026-03-16*
