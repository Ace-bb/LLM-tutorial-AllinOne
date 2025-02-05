
# Prompt tuning
Prompt Tuning 是一种软提示方法，出自论文《The Power of Scale for Parameter-Efficient Prompt Tuning》（[论文链接](https://arxiv.org/abs/2104.08691)）。它的目标是解决以下两个问题：

1. **全量微调的高开销和高成本**：传统方法需要对整个模型进行微调，这既耗时又耗费资源。
2. **人工设计提示词的成本高且效果不理想**：手动设计提示词不仅费时费力，效果还往往不尽如人意。

Prompt Tuning 的核心思想是：
- **学习提示词而非手动设计**：通过训练自动学习提示词，而不是依赖人工设计。
- **仅更新提示词部分的参数**：在训练过程中，只更新与提示词相关的参数，保持模型原始权重不变。

这样做的好处是：
- **模型复用性高**：同一个模型可以轻松应用于多个任务，无需重新训练整个模型。
- **效率大幅提升**：既节省了资源，又提高了模型的使用灵活性。


如图8所示，Prompt Tuning 的做法是为每个任务定义一组特定的 Prompt token（长度为 $k$），然后在输入层把这些 token 和数据拼接在一起作为输入。具体来说，就是把原本的输入 $X=\ [x_{1},\,x_{2},...,\,x_{m}]$ 变成 $X^{\prime}=\;[x_{1}^{\prime}$ $x_{2}^{\prime},...,\,x_{\mathrm{k}}^{\prime};\,x_{1},\,x_{2},...,\,x_{m}\big]$，然后通过公式 $\pmb{Y}=\pmb{W}\pmb{X}^{\prime}$ 进行计算。

这种方法有以下几个特点：
1. **模型参数不变**：整个预训练模型的参数保持不变，不需要重新训练。
2. **仅更新少量参数**：只允许在每个下游任务中更新额外的 $k$ 个 token。
3. **高效微调**：通过增加不到 $0.01\%$ 的任务特定参数，就可以微调超过10亿个参数的模型。

简单来说，Prompt Tuning 是一种非常高效的微调方法，既保留了预训练模型的强大能力，又只需要极少的额外参数就能适应新任务。

![](https://gitee.com/Ace_bb/static_resource_cloud/raw/master/LLMBook/LLMBOOK1/images/878d61ab49786bdb63e62b6883db529359e954d07a6f5cc27566d279a98ee2a5.jpg)
图8 Prompt Tuning 中任务特定token 与输入提示词拼接示意图

实验表明，随着预训练模型的参数量增加，`Prompt Tuning` 的效果会逐渐接近全量微调的效果。此外，`Prompt token` 的初始化方法和长度也会影响模型的性能。通过消融实验，我们发现：

1. **初始化方法的影响**：
   - 使用类标签初始化模型的效果比随机初始化或使用样本词汇表初始化更好。
   - 但随着模型参数规模的增加，这种优势会逐渐消失。

2. **Prompt token 长度的影响**：
   - 当 `Prompt token` 的长度在 20 左右时，模型的表现最好。
   - 不过，增加 `Prompt token` 的长度对模型性能的影响并不显著。

总结来说，模型参数量的增加会让 `Prompt Tuning` 的效果更接近全量微调，而 `Prompt token` 的初始化和长度虽然有一定影响，但随着模型规模的增大，这些影响会逐渐减弱。


总的来说，Prompt Tuning 通过一种端到端的方式，在连续且可微的参数空间里自动寻找合适的 prompt，取代了以前在离散空间里手动或自动设计提示词的做法。这种方法让 Prompt Tuning 在超大模型（比如 10B 级别的模型）上表现得和全量微调差不多好（虽然在小模型上效果稍微差一点）。

----

一句话总结什么事Prompt-tuning：
Prompt-tuning is an efficient, low-cost way of adapting an AI foundation model to new downstream tasks without retraining the model and updating its weights.\
**Prompt-tuning是一种高效的，低成本的，针对下游任务微调AI模型，而不需要重新训练模糊和更新模型参数的方法。**

Prompt-Tuning通过在输入层添加prompt tokens来为每个任务定制模型。 这些prompt tokens可以看作是模型的一种“提示”，它们被添加到输入序列的开头或结尾，以引导模型更好地适应不同任务。 与传统的微调方法相比，Prompt Tuning只需要调整一小部分参数，即prompt tokens，而无需对整个模型进行大规模的训练。

### 划重点：Prompt-tuning是只微调输入层增加的token参数，假设原本输入层维度是100，下一个隐藏层的维度是50，那么这部分的参数矩阵就是(100*50)，增加了Prompt token之后，输入层维度增加到120，那么这部分的参数矩阵就变成了(120*50)维，只调整增加的这20*50的参数量，其他的所有参数不改变。如图：

![Prompt-tuning](./img/promp-tuning.png)

相较于传统的微调方法，Prompt-Tuning具有以下优势：
1. 参数高效：Prompt-Tuning通过训练Prompt的权重来实现模型微调，而不是对整个模型进行训练。这大大减少了需要训练的参数数量，降低了计算复杂度，使得微调过程更加高效。
2. 灵活性高：Prompt-Tuning允许用户根据需要自定义Prompt，这使得微调过程更加灵活。用户可以根据任务需求、数据特点等因素，设计合适的Prompt来引导模型输出。
3. 易于实现：Prompt-Tuning的实现相对简单，不需要复杂的算法和计算资源。这使得Prompt-Tuning成为了一种易于普及和应用的微调方法。

## 实现步骤
在实际应用中，我们可以通过以下步骤来实施Prompt-Tuning：
1. 确定任务和目标：首先，我们需要明确模型需要完成的任务和目标。这将有助于我们设计合适的Prompt来引导模型的输出。
2. 设计Prompt：根据任务需求和数据特点，设计具有上下文的词或句子序列作为Prompt。Prompt的设计应充分考虑任务的语义信息和上下文关系。
3. 训练Prompt权重：在保持模型其他部分不变的情况下，仅训练Prompt的权重。这可以通过标准的梯度下降算法或其他优化算法来实现。
4. 评估和调整：在训练过程中，我们需要不断评估模型的性能，并根据评估结果对Prompt进行调整。这可以通过调整Prompt的长度、结构或语义信息等方式来实现。

通过以上步骤，我们可以实现Prompt-Tuning在模型微调过程中的应用。在实际操作中，我们还需要注意以下几点：
1. 注意Prompt的多样性和泛化能力：设计Prompt时，应尽量保证其在不同任务和数据集上的多样性和泛化能力，以提高模型的适应性和鲁棒性。
2. 结合其他技术：Prompt-Tuning可以与其他微调技术相结合，如知识蒸馏、迁移学习等。这可以进一步提高模型的性能和效率。
3. 注计算资源：虽然Prompt-Tuning在参数高效方面具有优势，但在实际应用中仍需关注计算资源的消耗。合理分配计算资源，以提高训练速度和效率。

## 源码实现
huggingface peft关于Prompt-tuning的核心代码实现在[PromptEmbedding](https://github.com/huggingface/peft/blob/main/src/peft/tuners/prompt_tuning/model.py):
```python
class PromptEmbedding(torch.nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt embedding.

    Example:

    ```py
    >>> from peft import PromptEmbedding, PromptTuningConfig

    >>> config = PromptTuningConfig(
    ...     peft_type="PROMPT_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     prompt_tuning_init="TEXT",
    ...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
    ...     tokenizer_name_or_path="t5-base",
    ... )

    >>> # t5_model.shared is the word embeddings of the base model
    >>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
    ```

    Input Shape: (`batch_size`, `total_virtual_tokens`)

    Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config, word_embeddings):
        super().__init__()

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)
        if config.prompt_tuning_init == PromptTuningInit.TEXT and not config.inference_mode:
            from transformers import AutoTokenizer

            tokenizer_kwargs = config.tokenizer_kwargs or {}
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, **tokenizer_kwargs)
            init_text = config.prompt_tuning_init_text
            init_token_ids = tokenizer(init_text)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]
            init_token_ids = torch.LongTensor(init_token_ids).to(word_embeddings.weight.device)
            with gather_params_ctx(word_embeddings.parameters()):
                word_embedding_weights = word_embeddings(init_token_ids).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings
```
## Prompt-tuning官方介绍
Google reasearch的Prompt tuning官方介绍：[https://github.com/google-research/prompt-tuning](https://github.com/google-research/prompt-tuning)

## 优质资源
本部分主要取自：
[大模型微调实践——Prompt tuning、PET、Prefix tuning、P-tuning的原理、区别与代码解析(一)](https://mp.weixin.qq.com/s?__biz=Mzg4MTkwMTQ4NA==&mid=2247484100&idx=1&sn=9a16611524c7953361717769284e7802&chksm=cf5fa807f8282111d1c8912ce305072f3e5558506d45b4d634d8e315cb84bdb7b5d92a7747c6&scene=21#wechat_redirect)

关于Prompt-tuning的综述，欢迎拜读华师数据学院·王嘉宁的文章(已收录)：\
[Prompt-Tuning——深度解读一种新的微调范式](https://blog.csdn.net/qq_36426650/article/details/120607050)

实操建议看苏神的代码：
[Pattern-Exploiting Training](https://github.com/bojone/Pattern-Exploiting-Training/tree/master)
苏神的解读原文链接：[必须要GPT3吗？不，BERT的MLM模型也能小样本学习](https://kexue.fm/archives/7764)
原文解读和代码均已收录于本目录中。\
sentiment.py是情感分类任务的代码，数据集下载链接：[情感分类数据集](https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip) \
代码的解读已在源码中进行详细标注。其中使用到的bert4keras原仓库链接为：[bert4karas](https://github.com/bojone/bert4keras)，是一个轻量级的keras版bert。

使用Prompt-tuning来微调大语言模型案例可看该目录下的：Prompt-Tuning代码实践.ipynb

至此，Prompt-Tuning完成。