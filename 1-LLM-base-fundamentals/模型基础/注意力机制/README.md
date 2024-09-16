# 注意力机制

注意力机制是一种在大语言模型中模拟人类注意力的技术，它通过动态调整输入数据的权重，使模型能够集中处理信息中最关键的部分。

## 1. 多头注意力
MHA（Multi-head Attention）是标准的多头注意力机制，包含h个Query、Key 和 Value 矩阵。所有注意力头的 Key 和 Value 矩阵权重不共享
Pytorch的代码如下：
```python
 fromtorch.nn.functionalimportscaled_dot_product_attention
 
 # shapes: (batch_size, seq_len, num_heads, head_dim)
 query=torch.randn(1, 256, 8, 64)
 key=torch.randn(1, 256, 8, 64)
 value=torch.randn(1, 256, 8, 64)
 
 output=scaled_dot_product_attention(query, key, value)
 print(output.shape) # torch.Size([1, 256, 8, 64])
```
对于每个查询头，都有一个对应的键。这个过程如下图所示:

![Alt text](multi_head_attention.png)

## 2. 稀疏注意力

![Alt text](xs_attention.png)

## 3. 滑动窗口注意力

![Alt text](slide_window_attention.png)

## 4. Multi-Query Attention
MQA（Multi-Query Attention，Fast Transformer Decoding: One Write-Head is All You Need）是多查询注意力的一种变体，也是用于自回归解码的一种注意力机制。与MHA不同的，MQA 让所有的头之间共享同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量。

## 5. Grouped-Query Attention
GQA（Grouped-Query Attention，GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints）是分组查询注意力，GQA将查询头分成G组，每个组共享一个Key 和 Value 矩阵。GQA-G是指具有G组的grouped-query attention。GQA-1具有单个组，因此具有单个Key 和 Value，等效于MQA。若GQA-H具有与头数相等的组，则其等效于MHA。

![Alt text](mha_gqa_mqa.png)