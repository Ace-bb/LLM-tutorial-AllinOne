# Adapter Tuning
Adapter Tuning 是一种适配器方法，最初在论文《Parameter-Efficient Transfer Learning for NLP》（[论文链接](https://arxiv.org/abs/1902.00751)）中提出。它的核心思想是：

1. **Adapter 层的添加**：在每个 Transformer 层中插入两个专门用于下游任务的 Adapter 层。
2. **参数更新策略**：只更新这些新添加的 Adapter 层的参数，而保持原预训练模型的参数不变（即“冻结”）。
3. **参数量增加**：这种方法大约会增加 $3.6\%$ 的参数量。
4. **算力开销减少**：与全量微调相比，Adapter Tuning 显著降低了训练时的计算资源消耗。

这种设计的优势在于：
- **模型扩展性**：当新的下游任务出现时，只需添加新的 Adapter 层即可，无需重新训练整个模型。
- **避免灾难性遗忘**：由于原预训练模型的参数保持不变，Adapter Tuning 有效避免了全量微调可能导致的灾难性遗忘问题。


Adapter 层集成到 Transformer 的具体结构如图3 所示。我们可以从图3 的左图看到 Adapter 层的结构，它由两个前馈子层组成：

1. 输入向量（维度为 $d$）首先通过第一个前馈子层（称为 `down-project`），将维度从 $d$ 降到 $m$。
2. 接着，通过一个非线性激活层。
3. 然后，通过第二个前馈子层（称为 `up-project`），将维度从 $m$ 升回原来的 $d$。

为了控制 Adapter 层的参数量，通常会让 $m$ 远小于 $d$。此外，Adapter 模块还通过`skip connection`的技术，将输入直接加到输出上。这样做的好处是，即使 Adapter 的参数初始值接近 0，输出也能接近恒等映射，从而保证训练的有效性。

再看图3 的右图，它展示了 Adapter 层如何集成到 Transformer 中：

- 在 Transformer 的每一层中，都会添加两个 Adapter 层：
  - 一个放在**多头注意力**（`multi-head attention`）之后。
  - 另一个放在**前馈神经网络**（`feed-forward network`）之后。

这样，Adapter 层就能在不显著增加模型参数的情况下，灵活地调整 Transformer 的行为。
![](https://gitee.com/Ace_bb/static_resource_cloud/raw/master/LLMBook/LLMBOOK1/images/fd85890f16c2090947c749c63e693d352d7f736f2f91f5f076134e57b8f9ba2f.jpg)
图3 Adapter 层的架构以及与Transformer 的集成

实验结果显示，`Adapter Tuning` 方法的效果和全量微调差不多，但 `Adapter` 中间层特征维度 $m$ 的最佳值会根据数据集的大小而变化。具体来说：

- 对于最小的 `RTE` 数据集，$m$ 的最佳值是 `8`。
- 对于 `MINI` 数据集，$m$ 的最佳值是 `256`。

不过，如果一直把 $m$ 固定为 `64`，平均准确率会稍微下降一些。


总的来说，Adapter Tuning 是一种通过添加少量额外参数（`$0.5\%\sim5\%_{\times}$`）就能让模型性能接近全量微调的方法，性能差距通常不超过 `$1\%$`。不过，这种方法也有一个明显的缺点：

- **模型层数增加**：添加 Adapter 后，模型的层数变多，导致训练和推理速度变慢。
- **计算资源消耗**：Adapter 层需要额外的计算资源。
- **通信开销增加**：在并行训练时，Adapter 层会产生额外的通信量，从而延长通信时间。

简单来说，虽然 Adapter Tuning 在性能上表现不错，但它会拖慢模型的速度，主要是因为增加了计算和通信的负担。


## Adapter Tuning的核心思想：
基于特征的迁移和微调的思想是将预训练模型抽象为$f(w)$，对下游任务微调抽象为$g(v, f(w))$，微调的过程是不断学习修改参数w和v，这样就导致预训练模型的参数$w$被修改，进而导致极高的训练成本。
Adapter Tuning的思想是将预训练模型抽象为$f(w)$，对下游任务微调抽象为$g(v, w)$，微调的过程是不断学习修改参数v，直接复用预训练模型的参数$w$而不是修改它，又因为参数$v$的数量级远小于参数$w$，因此训练成本极低。另外，针对新的下游任务n只需要增加新的Adapter，训练对应的参数$vn$。
$g(v, w)$的具体代码实现等效于，在原有预训练模型的网络结构中，插入一些Adapter层，预训练模型参数$w$作为Adapter层的入参，训练的目标是学习并修改Adapter层的参数$v$。

## Adapter Tuning(原理)
Adatper Tuning具体是如何实现的呢？论文中详细解释了Adapter层的网络结构，以及如何在原始的预训练模型上插入这些Adapter层：

- Adapter层的插入位置：在Transformer的多头注意力+前馈网络层之后，2x前馈网络层之后，分别插入了Adapter层。另外，在每个Adapter层之后还插入了一个Layer Norm层。

![Adapter tuning](https://jherculesqz.github.io/AI%E6%8B%BE%E9%81%97/%E3%80%90chatGPT%E3%80%91%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B043-LLM%E5%BE%AE%E8%B0%83%E6%8A%80%E6%9C%AF%E4%B9%8BAdapterTuning/image-20240322124748365.png)


- Adapter层的内部结构：Adapter层包含3层
前馈网络的向量降维层：用于将前一层预训练模型输出的高维向量，降维为低维向量。
非线性处理层：对下游任务微调时，学习参数v。
前馈网络的向量升维层：用于将Adapter层输出的低维向量，升维为高维向量。

- Adapter层的参数数量计算公式：$count(v)=2md+d+m$

    - $d$：前一层预训练模型输出的高维向量的维数。
    - m：Adapter层降维后的低维向量维数。
    - 实践经验：当m远小于d时，Adapter层的参数量会很小。论文给出的经验数据是可以通过控制m的数值，将Adapter层的参数量控制为预训练大模型参数量的0.5%~8%。这样，可以精准控制微调成本。

![Adapter层](https://jherculesqz.github.io/AI%E6%8B%BE%E9%81%97/%E3%80%90chatGPT%E3%80%91%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B043-LLM%E5%BE%AE%E8%B0%83%E6%8A%80%E6%9C%AF%E4%B9%8BAdapterTuning/image-20240322124814661.png)