# Adapter Tuning

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