# P-Tuning v2
P-Tuning v2 是 P-Tuning 的升级版本，来自论文《P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks》（[论文链接](https://arxiv.org/abs/2110.07602)）。它的主要目标是解决 P-Tuning 中存在的一些问题。

具体来说：
- P-Tuning v2 是对 P-Tuning 的改进。
- 它解决了 P-Tuning 的一些局限性。

总结一下，P-Tuning v2 是一个更强大的版本，旨在让 Prompt Tuning 在各种规模和任务中都能与 Fine-tuning 相媲美。

P-Tuning v2 是一种改进的微调方法，它借鉴了 Prefix-Tuning 的核心思想，但在实现上有所不同。具体来说：

1. **前缀参数的扩展**：
   - 在 Transformer 的每一层都增加了可微调的前缀参数。
   - 这种做法比 P-Tuning 只在第一层进行微调的方式更灵活，因为它引入了更多的可学习参数。

2. **重参数化编码器的调整**：
   - P-Tuning v2 的作者发现，重参数化的编码器（比如 Prefix-Tuning 中的 `MLP` 和 P-Tuning 中的 `LSTM`）可能会影响模型的效果。
   - 因此，在 P-Tuning v2 中，移除了这些重参数化的编码器，以提升模型的性能。

总结来说，P-Tuning v2 通过扩展前缀参数的范围并优化编码器设计，实现了更高效的微调效果。

与 `P-Tuning v1` 相比，`P-Tuning v2` 的一个关键改进是：**它将连续提示应用到了预训练模型的每一层，而不仅仅是输入层**。虽然 `P-Tuning v2` 和 `Prefix-Tuning` 方法看起来有点像，但它们的目标领域不同：

- `P-Tuning v2` 主要针对**自然语言理解**领域。
- `Prefix-Tuning` 则主要面向**文本生成**领域。

如图4 所示：
- 浅色部分表示**可训练的参数**。
- 深色部分表示**被冻结的参数**。

另外，针对不同任务，提示词的长度也会影响效果：
- 对于比较简单的分类任务，可以用**较短的提示词**（小于20 个 `token`）。
- 对于比较复杂的理解任务，则可以用**较长的提示词**（100 个左右 `token`）。

![](https://gitee.com/Ace_bb/static_resource_cloud/raw/master/LLMBook/LLMBOOK1/images/ed0632445ec178a685a32b67c14a7233f7db0a91075f7fd356380d9a04394d24.jpg)
图4 P-Tuning 与P-Tuning v2 对比

实验表明，即使在难度较大的序列标注任务上，P-Tuning v2 的表现也能与全量微调相媲美，而且它只需要微调总参数的 `$0.1\%\sim3\%$`。P-Tuning v2 可以看作是 Prefix-Tuning 的一个更通用的优化版本，具有以下特点：

- **更好的灵活性**：适应更多任务场景。
- **更少的可学习参数**：减少了训练开销。

此外，P-Tuning v2 在自然语言理解任务中表现出色，并且适用范围更广。

---

P-Tuning v2是P-Tuning的进一步改进版，在P-Tuning中，连续提示被插入到输入序列的嵌入层中，除了语言模型的输入层，其他层的提示嵌入都来自于上一层。这种设计存在两个问题：

- 第一，它限制了优化参数的数量。由于模型的输入文本长度是固定的，通常为512，因此提示的长度不能过长。
- 第二，当模型层数很深时，微调时模型的稳定性难以保证；模型层数越深，第一层输入的提示对后面层的影响难以预测，这会影响模型的稳定性。

P-Tuning v2的改进在于，不仅在第一层插入连续提示，而是在多层都插入连续提示，且层与层之间的连续提示是相互独立的。这样，在模型微调时，可训练的参数量增加了，P-Tuning v2在应对复杂的自然语言理解(NLU)任务和小型模型方面，相比原始P-Tuning具有更出色的效能。

除了以上PEFT，当前还存在PILL（Pluggable Instruction Language Learning）、SSF（Scaling & Shifting Your Features）等其他类型的微调方法。

PILL是PEFT的一个特定实现，特别关注于如何通过插入可训练的模块或插件来提升模型的任务适应性。这些插件被设计为与原始模型协同工作，以提高模型在处理特定任务时的效率和效果。

SSF核心思想是对模型的特征（即模型层的输出）进行缩放（Scaling）和位移（Shifting）。简单来说，就是通过调整特征的比例和偏移量来优化模型的性能。

这种方法可以在改善模型对特定任务的响应时，不需要调整或重新训练模型中的所有参数，从而在节省计算资源的同时保持或提升模型性能。这对于处理大规模模型特别有效，因为它减少了训练和调整所需的资源和时间。