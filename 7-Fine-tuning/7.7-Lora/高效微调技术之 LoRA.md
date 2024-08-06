## 前言
在自然语言处理领域，大[语言模型](https://so.csdn.net/so/search?q=%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&spm=1001.2101.3001.7020)的预训练 - 微调技术已经成为一种常见的方法。其中，LoRA（Low-Rank Adaptation）是一种新颖的微调技术，通过引入低秩矩阵来调整模型的行为，以提高模型在新任务上的表现。本文将对 LoRA 的原理、优势以及应用进行详细介绍。
## 一、微调技术分类
微调技术主要分为以下几类：
**1）增加额外参数（A**）：这种方法是在原有的[预训练模型](https://so.csdn.net/so/search?q=%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B&spm=1001.2101.3001.7020)的基础上增加一些额外的参数，以改变模型的行为。
**2）选取一部分参数更新（S）**：这种方法是在微调过程中只更新模型的一部分参数，而不是所有参数。这可以减少计算量，提高微调效率。
**3）引入重参数化（R）**：这种方法是在模型的参数空间中引入一些新的变化，通常是一些线性变换或非线性变换，以改变模型的行为。这种方法可以使模型在新任务上有更好的表现。
> 常见的参数**高效微调技术**有 **Prefix Tuning、Prompt Tuning、P-Tuning、Adapter Tuning、LoRA** 等

## 二、LoRA 原理
LoRA（Low-Rank Adaptation: 低秩的适配器）是一种新颖的微调技术，它通过引入低秩矩阵来调整模型的行为，以提高模型在新任务上的表现。具体来说，**LoRA 在原有的预训练模型中增加了两个旁路矩阵 A 和 B，这两个矩阵的维度远小于原始模型的输入输出维度，从而实现了参数的高效微调**，从而减少适配下游任务所需要训练的参数。给定一个参数矩阵 $\mathbf W$，其 更新过程可以一般性地表达为以下形式：
$\mathbf W= \mathbf W_0+ \Delta\mathbf W$
其中，$\mathbf W_0$ 是原始参数矩阵，$\Delta\mathbf W$ 是更新的梯度矩阵。LoRA 的基本思想是冻结原 始矩阵 $\mathbf W_0 ∈ R^{H*H}$，通过低秩分解矩阵 $\mathbf A ∈ R^{H*H}$和  $\mathbf B ∈ R^{H*H}$ 来近似参数更新矩阵 $\Delta W=A\cdot B^T$，其中 $R << H$ 是减小后的秩。在微调期间，原始的矩阵参数 $W_0$不会被更新，低秩分解矩阵 $\mathbf A$ 和 $\mathbf B$则是可训练参数用于适配下游任务。在前向传 播过程中，原始计算中间状态 $\mathbf h = \mathbf W_0 \cdot \mathbf x$ 的公式修改为:
$\mathbf h = \mathbf W_0 \cdot x + \mathbf A \cdot \mathbf B^T \cdot x$
在训练完成后，进一步将原始参数矩阵 $\mathbf W_0$ 和训练得到的权重  $\mathbf A$ 和 $\mathbf B$ 进行合并：$\mathbf W = \mathbf W_0 + \mathbf A \cdot \mathbf B^T$，得到更新后的参数矩阵。因此，LoRA 微调得到的模型在解码过 程中不会增加额外的开销。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/1805392/1722495623736-08790eab-5fca-4c4c-8750-edee206f9747.png#averageHue=%23f6f5f3&clientId=u8cd6d09a-cbc3-4&from=paste&height=253&id=uf603d980&originHeight=692&originWidth=1328&originalType=binary&ratio=2&rotation=0&showTitle=false&size=242760&status=done&style=none&taskId=u5fce9f0a-7d25-414e-bb6b-a9dbc7a4b99&title=&width=486)
![](https://img-blog.csdnimg.cn/direct/80c422fe5fae4cef91f9f978fef0c852.png#id=xdyeh&originHeight=472&originWidth=1249&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
## 三、在哪儿增加旁路
在原有的预训练模型中，可以选择在**任意两个相邻层之间增加旁路矩阵 A 和 B**。这样，模型在前向传播过程中，可以通过这两个旁路矩阵来引入新的信息，从而改变模型的行为。
![](https://img-blog.csdnimg.cn/direct/bab6eeebdd944e0f8f36c4297a63f93a.png#id=W6Q2J&originHeight=679&originWidth=1185&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
## 四、为什么微调少量参数就可以
![](https://img-blog.csdnimg.cn/direct/c6995b5d3ed54d069b382cf2e86942c9.png#id=RbIup&originHeight=270&originWidth=601&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
**A 的输入维度和 B 的输出维度分别与原始模型的输入输出维度相同**，而 **A 的输出维度和 B 的输入维度是一个远小于原始模型输入输出维度的值，这就是 low-rank 的体现**，可以极大地减少待训练的参数
![](https://img-blog.csdnimg.cn/direct/880c027ff3bb413db770025d1d2446ae.png#id=mGJHr&originHeight=198&originWidth=604&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
**秩**表示的是矩阵的信息量，这里的 “**秩**” 特指引入的旁路矩阵的规模，即它们的行数和列数。
![](https://img-blog.csdnimg.cn/direct/019e79d56d644c4ab8c7b4fa4fdd4b68.png#id=TTPgf&originHeight=120&originWidth=664&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
在 LoRA 技术中，我们通过引入低秩矩阵来调整预训练模型的行为，同时保留大部分原有的参数不变。这样做可以在不牺牲太多性能的前提下，**显著降低模型微调时的计算成本和内存需求**。
> **通俗化解释：“秩”**:
想象一下你有一个很大的包裹，你需要通过一个小门把它送出去。但是门太小了，你必须把包裹拆成几个小包裹才能通过。在这个比喻中，大包裹就像模型的权重矩阵，小门就像我们新增的低秩矩阵，而 “秩” 就是这些小包裹的数量。在 LoRA 中，我们通过创建一些小的（低秩）矩阵来传递信息，而不是使用原始的大矩阵。这样做的好处是我们可以只关注那些最重要的信息，忽略掉不重要的信息，从而减少计算量和内存需求。

## 五、如何对 A 和 B 进行初始化
#### A 和 B 如何初始化？
对 A 采用高斯初始化，对 B 采用零初始化的目的是，让训练刚开始时的值为 0，这样不会给模型带来额外的噪声。
![](https://img-blog.csdnimg.cn/direct/db487699d0594b94b7de2dac4bd3dc8b.png#id=CotUq&originHeight=688&originWidth=526&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
## 六、增加旁路会增加推理时间吗？
虽然增加了旁路矩阵 A 和 B，但是由于它们的维度远小于原始模型的输入输出维度，因此在推理过程中，计算量的增加是非常有限的。
![](https://img-blog.csdnimg.cn/direct/26a977c7b2454395802b2b5951dc9a20.png#id=G2wgg&originHeight=619&originWidth=1201&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
## 七、R 值为多少合适
**R 值表示的是旁路矩阵 A 和 B 的秩。一般来说，R 值的选择需要根据具体任务和模型结构来确定**。在实际应用中，可以尝试不同的 R 值，以找到最佳的设置。
![](https://img-blog.csdnimg.cn/direct/1f93b6ea77c140b1be084db71a33a7c8.png#id=QQfAV&originHeight=690&originWidth=658&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
## 八、如何注入 LoRA
要将 LoRA 应用于现有的预训练模型中，首先需要在相邻层之间插入旁路矩阵 A 和 B。然后，在微调过程中，只需要调整这两个旁路矩阵的参数即可。这样，就可以实现模型行为的高效调整。
![](https://img-blog.csdnimg.cn/direct/e248f1105e694cea920791f6c8bc2fc2.png#id=btHSZ&originHeight=369&originWidth=544&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
如上图中定义一个简单的 3 层的神经网络，在第 1 层增加旁路后效果如下：
![](https://img-blog.csdnimg.cn/direct/fbcd6d26ea964ded92ac886a60083fae.png#id=bjpeT&originHeight=625&originWidth=682&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
## 九、LoRA 代码实践
PEFT 文档资料地址
1）文档地址：[https://huggingface.co/docs/peft/index](https://huggingface.co/docs/peft/index)
2）Github 地址：[https://github.com/huggingface/peft](https://github.com/huggingface/peft)
[PEFT](https://huggingface.co/docs/peft/index)（Parameter-Efficient Fine-Tuning）库是一个用于参数高效微调预训练语言模型的库，旨在降低大规模模型微调的计算和存储成本。
PEFT 库的核心优势在于它能够仅通过微调少量额外模型参数来适应各种下游任务，避免了对整个大模型参数进行微调的需求。这种方法不仅降低了资源消耗，而且在很多情况下能达到与完全微调相当的性能
![](https://img-blog.csdnimg.cn/direct/ae7e3c00d83e4e2da59634cde43b7b2c.png#id=DV6EB&originHeight=555&originWidth=1465&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
PEFT 技术的支持：
![](https://img-blog.csdnimg.cn/direct/e52af848a4cf43efb7b78d33f985b84c.png#id=KvJE9&originHeight=526&originWidth=1283&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
### 学术资源加速
方便从 huggingface 下载模型，这云平台 [autodl](https://www.autodl.com/) 提供的，仅适用于 autodl。
```python
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

```
### 步骤 1 导入相关包
开始之前，我们需要导入适用于模型训练和推理的必要库，如 transformers。
```python
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
```
### 步骤 2 加载数据集
使用适当的数据加载器，例如 datasets 库，来加载预处理过的指令遵循性任务数据集。
```python
ds = Dataset.load_from_disk("/root/tuning/lesson01/data/alpaca_data_zh/")
ds
```
输出
```python
Dataset({
    features: ['output', 'input', 'instruction'],
    num_rows: 26858
})
```
数据查看
```
ds[:1]
```
输出
```
{'output': ['以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。'],
 'input': [''],
 'instruction': ['保持健康的三个提示。']}

```
### 步骤 3 数据集预处理
利用预训练模型的分词器（Tokenizer）对原始文本进行编码，并生成相应的输入 ID、注意力掩码和标签。
1）获取分词器
```python
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
tokenizer
```
![](https://img-blog.csdnimg.cn/direct/1f1c4e48721d42728b2ac4814f20db5b.png#id=lckq7&originHeight=140&originWidth=1090&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none)
输出：
```
BloomTokenizerFast(name_or_path='Langboat/bloom-1b4-zh', vocab_size=46145, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
    0: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    1: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    3: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}

```
2）定义数据处理函数
```python
def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

```
3）对数据进行预处理
```python
tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
tokenized_ds
```
输出：
```
Dataset({
    features: ['input_ids', 'attention_mask', 'labels'],
    num_rows: 26858
})
```
### 步骤 4 创建模型
然后，我们实例化一个预训练模型，这个模型将作为微调的基础。对于大型模型，我们可能还需要进行一些特定的配置，以适应可用的计算资源。
```python
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh", low_cpu_mem_usage=True)
```
查看总共有哪些层，可以基于这些层添加 LoRA
```python
for name, parameter in model.named_parameters():
    print(name)
```
输出
```
base_model.model.transformer.word_embeddings.weight
base_model.model.transformer.word_embeddings_layernorm.weight
base_model.model.transformer.word_embeddings_layernorm.bias
base_model.model.transformer.h.0.input_layernorm.weight
base_model.model.transformer.h.0.input_layernorm.bias
base_model.model.transformer.h.0.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.0.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.0.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.0.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.0.self_attention.dense.weight
base_model.model.transformer.h.0.self_attention.dense.bias
base_model.model.transformer.h.0.post_attention_layernorm.weight
base_model.model.transformer.h.0.post_attention_layernorm.bias
base_model.model.transformer.h.0.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.0.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.0.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.0.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.1.input_layernorm.weight
base_model.model.transformer.h.1.input_layernorm.bias
base_model.model.transformer.h.1.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.1.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.1.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.1.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.1.self_attention.dense.weight
base_model.model.transformer.h.1.self_attention.dense.bias
base_model.model.transformer.h.1.post_attention_layernorm.weight
base_model.model.transformer.h.1.post_attention_layernorm.bias
base_model.model.transformer.h.1.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.1.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.1.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.1.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.2.input_layernorm.weight
base_model.model.transformer.h.2.input_layernorm.bias
base_model.model.transformer.h.2.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.2.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.2.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.2.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.2.self_attention.dense.weight
base_model.model.transformer.h.2.self_attention.dense.bias
base_model.model.transformer.h.2.post_attention_layernorm.weight
base_model.model.transformer.h.2.post_attention_layernorm.bias
base_model.model.transformer.h.2.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.2.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.2.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.2.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.3.input_layernorm.weight
base_model.model.transformer.h.3.input_layernorm.bias
base_model.model.transformer.h.3.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.3.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.3.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.3.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.3.self_attention.dense.weight
base_model.model.transformer.h.3.self_attention.dense.bias
base_model.model.transformer.h.3.post_attention_layernorm.weight
base_model.model.transformer.h.3.post_attention_layernorm.bias
base_model.model.transformer.h.3.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.3.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.3.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.3.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.4.input_layernorm.weight
base_model.model.transformer.h.4.input_layernorm.bias
base_model.model.transformer.h.4.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.4.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.4.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.4.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.4.self_attention.dense.weight
base_model.model.transformer.h.4.self_attention.dense.bias
base_model.model.transformer.h.4.post_attention_layernorm.weight
base_model.model.transformer.h.4.post_attention_layernorm.bias
base_model.model.transformer.h.4.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.4.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.4.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.4.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.5.input_layernorm.weight
base_model.model.transformer.h.5.input_layernorm.bias
base_model.model.transformer.h.5.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.5.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.5.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.5.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.5.self_attention.dense.weight
base_model.model.transformer.h.5.self_attention.dense.bias
base_model.model.transformer.h.5.post_attention_layernorm.weight
base_model.model.transformer.h.5.post_attention_layernorm.bias
base_model.model.transformer.h.5.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.5.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.5.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.5.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.6.input_layernorm.weight
base_model.model.transformer.h.6.input_layernorm.bias
base_model.model.transformer.h.6.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.6.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.6.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.6.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.6.self_attention.dense.weight
base_model.model.transformer.h.6.self_attention.dense.bias
base_model.model.transformer.h.6.post_attention_layernorm.weight
base_model.model.transformer.h.6.post_attention_layernorm.bias
base_model.model.transformer.h.6.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.6.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.6.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.6.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.7.input_layernorm.weight
base_model.model.transformer.h.7.input_layernorm.bias
base_model.model.transformer.h.7.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.7.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.7.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.7.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.7.self_attention.dense.weight
base_model.model.transformer.h.7.self_attention.dense.bias
base_model.model.transformer.h.7.post_attention_layernorm.weight
base_model.model.transformer.h.7.post_attention_layernorm.bias
base_model.model.transformer.h.7.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.7.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.7.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.7.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.8.input_layernorm.weight
base_model.model.transformer.h.8.input_layernorm.bias
base_model.model.transformer.h.8.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.8.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.8.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.8.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.8.self_attention.dense.weight
base_model.model.transformer.h.8.self_attention.dense.bias
base_model.model.transformer.h.8.post_attention_layernorm.weight
base_model.model.transformer.h.8.post_attention_layernorm.bias
base_model.model.transformer.h.8.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.8.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.8.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.8.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.9.input_layernorm.weight
base_model.model.transformer.h.9.input_layernorm.bias
base_model.model.transformer.h.9.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.9.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.9.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.9.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.9.self_attention.dense.weight
base_model.model.transformer.h.9.self_attention.dense.bias
base_model.model.transformer.h.9.post_attention_layernorm.weight
base_model.model.transformer.h.9.post_attention_layernorm.bias
base_model.model.transformer.h.9.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.9.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.9.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.9.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.10.input_layernorm.weight
base_model.model.transformer.h.10.input_layernorm.bias
base_model.model.transformer.h.10.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.10.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.10.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.10.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.10.self_attention.dense.weight
base_model.model.transformer.h.10.self_attention.dense.bias
base_model.model.transformer.h.10.post_attention_layernorm.weight
base_model.model.transformer.h.10.post_attention_layernorm.bias
base_model.model.transformer.h.10.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.10.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.10.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.10.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.11.input_layernorm.weight
base_model.model.transformer.h.11.input_layernorm.bias
base_model.model.transformer.h.11.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.11.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.11.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.11.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.11.self_attention.dense.weight
base_model.model.transformer.h.11.self_attention.dense.bias
base_model.model.transformer.h.11.post_attention_layernorm.weight
base_model.model.transformer.h.11.post_attention_layernorm.bias
base_model.model.transformer.h.11.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.11.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.11.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.11.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.12.input_layernorm.weight
base_model.model.transformer.h.12.input_layernorm.bias
base_model.model.transformer.h.12.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.12.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.12.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.12.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.12.self_attention.dense.weight
base_model.model.transformer.h.12.self_attention.dense.bias
base_model.model.transformer.h.12.post_attention_layernorm.weight
base_model.model.transformer.h.12.post_attention_layernorm.bias
base_model.model.transformer.h.12.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.12.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.12.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.12.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.13.input_layernorm.weight
base_model.model.transformer.h.13.input_layernorm.bias
base_model.model.transformer.h.13.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.13.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.13.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.13.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.13.self_attention.dense.weight
base_model.model.transformer.h.13.self_attention.dense.bias
base_model.model.transformer.h.13.post_attention_layernorm.weight
base_model.model.transformer.h.13.post_attention_layernorm.bias
base_model.model.transformer.h.13.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.13.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.13.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.13.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.14.input_layernorm.weight
base_model.model.transformer.h.14.input_layernorm.bias
base_model.model.transformer.h.14.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.14.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.14.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.14.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.14.self_attention.dense.weight
base_model.model.transformer.h.14.self_attention.dense.bias
base_model.model.transformer.h.14.post_attention_layernorm.weight
base_model.model.transformer.h.14.post_attention_layernorm.bias
base_model.model.transformer.h.14.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.14.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.14.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.14.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.15.input_layernorm.weight
base_model.model.transformer.h.15.input_layernorm.bias
base_model.model.transformer.h.15.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.15.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.15.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.15.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.15.self_attention.dense.weight
base_model.model.transformer.h.15.self_attention.dense.bias
base_model.model.transformer.h.15.post_attention_layernorm.weight
base_model.model.transformer.h.15.post_attention_layernorm.bias
base_model.model.transformer.h.15.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.15.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.15.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.15.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.16.input_layernorm.weight
base_model.model.transformer.h.16.input_layernorm.bias
base_model.model.transformer.h.16.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.16.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.16.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.16.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.16.self_attention.dense.weight
base_model.model.transformer.h.16.self_attention.dense.bias
base_model.model.transformer.h.16.post_attention_layernorm.weight
base_model.model.transformer.h.16.post_attention_layernorm.bias
base_model.model.transformer.h.16.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.16.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.16.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.16.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.17.input_layernorm.weight
base_model.model.transformer.h.17.input_layernorm.bias
base_model.model.transformer.h.17.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.17.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.17.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.17.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.17.self_attention.dense.weight
base_model.model.transformer.h.17.self_attention.dense.bias
base_model.model.transformer.h.17.post_attention_layernorm.weight
base_model.model.transformer.h.17.post_attention_layernorm.bias
base_model.model.transformer.h.17.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.17.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.17.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.17.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.18.input_layernorm.weight
base_model.model.transformer.h.18.input_layernorm.bias
base_model.model.transformer.h.18.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.18.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.18.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.18.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.18.self_attention.dense.weight
base_model.model.transformer.h.18.self_attention.dense.bias
base_model.model.transformer.h.18.post_attention_layernorm.weight
base_model.model.transformer.h.18.post_attention_layernorm.bias
base_model.model.transformer.h.18.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.18.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.18.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.18.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.19.input_layernorm.weight
base_model.model.transformer.h.19.input_layernorm.bias
base_model.model.transformer.h.19.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.19.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.19.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.19.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.19.self_attention.dense.weight
base_model.model.transformer.h.19.self_attention.dense.bias
base_model.model.transformer.h.19.post_attention_layernorm.weight
base_model.model.transformer.h.19.post_attention_layernorm.bias
base_model.model.transformer.h.19.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.19.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.19.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.19.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.20.input_layernorm.weight
base_model.model.transformer.h.20.input_layernorm.bias
base_model.model.transformer.h.20.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.20.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.20.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.20.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.20.self_attention.dense.weight
base_model.model.transformer.h.20.self_attention.dense.bias
base_model.model.transformer.h.20.post_attention_layernorm.weight
base_model.model.transformer.h.20.post_attention_layernorm.bias
base_model.model.transformer.h.20.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.20.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.20.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.20.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.21.input_layernorm.weight
base_model.model.transformer.h.21.input_layernorm.bias
base_model.model.transformer.h.21.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.21.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.21.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.21.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.21.self_attention.dense.weight
base_model.model.transformer.h.21.self_attention.dense.bias
base_model.model.transformer.h.21.post_attention_layernorm.weight
base_model.model.transformer.h.21.post_attention_layernorm.bias
base_model.model.transformer.h.21.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.21.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.21.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.21.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.22.input_layernorm.weight
base_model.model.transformer.h.22.input_layernorm.bias
base_model.model.transformer.h.22.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.22.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.22.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.22.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.22.self_attention.dense.weight
base_model.model.transformer.h.22.self_attention.dense.bias
base_model.model.transformer.h.22.post_attention_layernorm.weight
base_model.model.transformer.h.22.post_attention_layernorm.bias
base_model.model.transformer.h.22.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.22.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.22.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.22.mlp.dense_4h_to_h.bias
base_model.model.transformer.h.23.input_layernorm.weight
base_model.model.transformer.h.23.input_layernorm.bias
base_model.model.transformer.h.23.self_attention.query_key_value.base_layer.weight
base_model.model.transformer.h.23.self_attention.query_key_value.base_layer.bias
base_model.model.transformer.h.23.self_attention.query_key_value.lora_A.default.weight
base_model.model.transformer.h.23.self_attention.query_key_value.lora_B.default.weight
base_model.model.transformer.h.23.self_attention.dense.weight
base_model.model.transformer.h.23.self_attention.dense.bias
base_model.model.transformer.h.23.post_attention_layernorm.weight
base_model.model.transformer.h.23.post_attention_layernorm.bias
base_model.model.transformer.h.23.mlp.dense_h_to_4h.weight
base_model.model.transformer.h.23.mlp.dense_h_to_4h.bias
base_model.model.transformer.h.23.mlp.dense_4h_to_h.weight
base_model.model.transformer.h.23.mlp.dense_4h_to_h.bias
base_model.model.transformer.ln_f.weight
base_model.model.transformer.ln_f.bias

```
#### LoRA 相关的配置
**（下面 2 个部分是 LoRA 相关的配置，其他的和全量微调代码一样）。**
#### 1、PEFT 步骤 1 配置文件
在使用 PEFT 进行微调时，我们首先需要创建一个配置文件，该文件定义了微调过程中的各种设置，如学习率调度、优化器选择等。
```python
from peft import LoraConfig, TaskType, get_peft_model
config = LoraConfig(task_type=TaskType.CAUSAL_LM)

config
```
#### 2、PEFT 步骤 2 创建模型
接下来，我们使用 PEFT 和预训练模型来创建一个微调模型。这个模型将包含原始的预训练模型以及由 PEFT 引入的低秩参数。
```python
model = get_peft_model(model, config)
model
```
输出：
```
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): PeftModelForCausalLM(
      (base_model): LoraModel(
        (model): BloomForCausalLM(
          (transformer): BloomModel(
            (word_embeddings): Embedding(46145, 2048)
            (word_embeddings_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (h): ModuleList(
              (0-23): 24 x BloomBlock(
                (input_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
                (self_attention): BloomAttention(
                  (query_key_value): lora.Linear(
                    (base_layer): Linear(in_features=2048, out_features=6144, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Identity()
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=2048, out_features=8, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=8, out_features=6144, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                  )
                  (dense): Linear(in_features=2048, out_features=2048, bias=True)
                  (attention_dropout): Dropout(p=0.0, inplace=False)
                )
                (post_attention_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
                (mlp): BloomMLP(
                  (dense_h_to_4h): Linear(in_features=2048, out_features=8192, bias=True)
                  (gelu_impl): BloomGelu()
                  (dense_4h_to_h): lora.Linear(
                    (base_layer): Linear(in_features=8192, out_features=2048, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Identity()
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=8192, out_features=8, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=8, out_features=2048, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                  )
                )
              )
            )
            (ln_f): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (lm_head): Linear(in_features=2048, out_features=46145, bias=False)
        )
      )
    )
  )
)

```
查看配置
```
config
```
输出
```
LoraConfig(peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path=None, revision=None, task_type=<TaskType.CAUSAL_LM: 'CAUSAL_LM'>, inference_mode=False, r=8, target_modules={'query_key_value', 'dense_4h_to_h'}, lora_alpha=8, lora_dropout=0.0, fan_in_fan_out=False, bias='none', modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', loftq_config={})
```
### 步骤 5 配置训练参数
定义训练参数，包括输出目录、学习率、批次大小、梯度累积步数、优化器选择等。
```python
args = TrainingArguments(
    output_dir="/root/autodl-tmp/tuningdata/lora",
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=8,
    logging_steps=20, 
    num_train_epochs=4 
)
```
### 步骤 6 创建训练器
最后，我们创建一个训练器实例，它封装了训练循环。训练器将负责运行训练过程，并根据我们之前定义的参数进行优化。
```python
trainer = Trainer(
    model=model,
    args=args, 
    train_dataset=tokenized_ds, 
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True) 
)
```
### 步骤 7 模型训练
通过调用训练器的`train()`方法，我们启动模型的训练过程。
```python
trainer.train()
```
### 步骤 8 模型推理
训练完成后，我们可以使用训练好的模型进行推理。这通常涉及到使用模型的`inference`方法，输入经过适当处理的问题，并得到模型的输出。
```python
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

ipt = "Human: {}\n{}".format("如何写好一个简历？", "").strip() + "\n\nAssistant: "
pipe(ipt, max_length=256, do_sample=True, )
```
输出
```
[{'generated_text': 'Human: 如何写好一个简历？\n\nAssistant: 一篇好的简历应包含以下内容：个人信息（姓名，出生日期，出生地，教育经历，工作经历）、求职理由、个人能力（如语言能力，英语水平，操作技能，编程能力，市场营销能力，分析归纳能力等）、学习经历、实践经历和经验、荣誉奖项、相关证书和荣誉、个人兴趣爱好以及在工作中遇到的瓶颈和障碍。\n\n在书写时，应注意文字简洁、条理清晰，突出重点，语言流畅。您也可以在简历中附上一些相关的个人照片或照片资料以供他人参考。如果您有任何疑问，请随时与我联系。'}]

```
## 十、主路合并旁路
### 1、加载基础模型
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh", low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
```
### 2、加载 LoRA 模型
```python
p_model = PeftModel.from_pretrained(model, model_id="/root/autodl-tmp/tuningdata/lora/checkpoint-500")
p_model
```
输出
```
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): BloomForCausalLM(
      (transformer): BloomModel(
        (word_embeddings): Embedding(46145, 2048)
        (word_embeddings_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (h): ModuleList(
          (0-23): 24 x BloomBlock(
            (input_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (self_attention): BloomAttention(
              (query_key_value): lora.Linear(
                (base_layer): Linear(in_features=2048, out_features=6144, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=6144, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (dense): Linear(in_features=2048, out_features=2048, bias=True)
              (attention_dropout): Dropout(p=0.0, inplace=False)
            )
            (post_attention_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (mlp): BloomMLP(
              (dense_h_to_4h): Linear(in_features=2048, out_features=8192, bias=True)
              (gelu_impl): BloomGelu()
              (dense_4h_to_h): lora.Linear(
                (base_layer): Linear(in_features=8192, out_features=2048, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=8192, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=2048, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
            )
          )
        )
        (ln_f): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=2048, out_features=46145, bias=False)
    )
  )
)

```
### 3、模型推理
```python
from transformers import pipeline

pipe = pipeline("text-generation", model=p_model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("如何写好一个简历？", "").strip() + "\n\nAssistant: "
pipe(ipt, max_length=256, do_sample=True, )

```
### 4、模型合并
```python
merge_model = p_model.merge_and_unload()
merge_model
```
输出
```
BloomForCausalLM(
  (transformer): BloomModel(
    (word_embeddings): Embedding(46145, 2048)
    (word_embeddings_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
    (h): ModuleList(
      (0-23): 24 x BloomBlock(
        (input_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (self_attention): BloomAttention(
          (query_key_value): Linear(in_features=2048, out_features=6144, bias=True)
          (dense): Linear(in_features=2048, out_features=2048, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (mlp): BloomMLP(
          (dense_h_to_4h): Linear(in_features=2048, out_features=8192, bias=True)
          (gelu_impl): BloomGelu()
          (dense_4h_to_h): Linear(in_features=8192, out_features=2048, bias=True)
        )
      )
    )
    (ln_f): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2048, out_features=46145, bias=False)
)

```
### 5、模型推理
```python
from transformers import pipeline

pipe = pipeline("text-generation", model=merge_model, tokenizer=tokenizer, device=0)
ipt = "Human:如何写好一个简历？\n\nAssistant: "
pipe(ipt, max_length=256,)
```
### 6、完整模型保存
模型训练完后，可以将合并的模型进行保存到本地，进行备用
```python
merge_model.save_pretrained("/root/autodl-tmp/tuningdata/merge_model")
```
## 总结
LoRA 是一种新颖的微调技术，通过引入低秩矩阵来调整模型的行为，以提高模型在新任务上的表现。它具有参数高效、计算复杂度低等优点，因此在自然语言处理领域具有广泛的应用前景。
