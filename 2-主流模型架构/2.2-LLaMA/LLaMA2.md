
​一、引言
-----

Llama 2 是 Meta 在 LLaMA 基础上升级的一系列从 7B 到 70B 参数的大语言模型。Llama2 在各个榜单上精度全面超过 LLaMA1，Llama 2 作为开源界表现最好的模型之一，目前被广泛使用。

为了更深入地理解 Llama 2 的技术特点，特地在此整理了 Llama 2 模型架构、 预训练、SFT、RLHF 内容详解，也从安全性角度进行了分析。

话不多说，直接上干货啦

一、LLaMA 2 简介
------------

论文：[https://arxiv.org/abs/2307.09288](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2307.09288)

Github：[GitHub - facebookresearch/llama: Inference code for LLaMA models](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/llama)

Meta 在原本的 LLaMA 1 的基础上，增加了预训练使用的 token 数量；同时，修改了模型的架构，引入了 Group Query Attention（GQA）。

并且，在 Llama 2 的基础上，Meta 同时发布了 Llama 2-Chat。其通过应用监督微调来创建 Llama 2-Chat 的初始版本。随后，使用带有人类反馈 (RLHF) 方法的强化学习迭代地改进模型，过程中使用了拒绝采样和近端策略优化 (PPO)。

Llama 2-Chat 的训练主要流程如下：

![](https://pic1.zhimg.com/v2-8dbde9ff25df67cf30adb665252136a4_r.jpg)

二、模型架构
------

### 2.1 主要架构

Llama 2 采用 LLaMA 1 的大部分预训练设置和模型架构，包括：

1.  **Tokenzier:** 和 LLaMA 1 一样的 tokenizer，使用 SentencePiece 实现的 BPE 算法。与 LLaMA 1 一样，将所有数字拆分为单个数字并使用字节来分解未知的 UTF-8 字符。总词汇量为 32k 个 token。  
    （关于 BPE，可参考 [BPE 算法原理及使用指南【深入浅出】 - 知乎](https://zhuanlan.zhihu.com/p/448147465)）
2.  **Pre-normalization**：为了提高训练稳定性，LLaMa 对每个 Transformer 子层的输入进行归一化，而不是对输出进行归一化。LLaMa 使用了 RMSNorm 归一化函数。  
    （关于 Pre-norm vs Post-norm，可参考[为什么 Pre Norm 的效果不如 Post Norm？ - 科学空间 | Scientific Spaces](https://link.zhihu.com/?target=https%3A//kexue.fm/archives/9009)）
3.  **SwiGLU 激活函数：**LLaMa 使用 SwiGLU 激活函数替换 ReLU 以提高性能，维度从 $4d$4d 变为 $\frac{2}{3}4d$\frac{2}{3}4d ​。SwiGLU 是一种激活函数，它是 GLU 的一种变体， 它可以提高 transformer 模型的性能。SwiGLU 的优点是它可以动态地调整信息流的门控程度，根据输入的不同而变化，而且 SwiGLU 比 ReLU 更`平滑`，可以带来`更好的优化`和`更快的收敛`。  
    （关于 SwiGLU 激活函数，可参考[激活函数总结（八）：基于 Gate mechanism 机制的激活函数补充 (GLU、SwiGLU、GTU、Bilinear、ReGLU、GEGLU)_glu 激活 - CSDN 博客](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_36758270/article/details/132174106)）  
    
4.  **Rotary Embeddings：**LLaMa 没有使用之前的绝对位置编码，而是使用了旋转位置编码（RoPE），可以提升模型的外推性。它的基本思想是通过一个旋转矩阵来调整每个单词或标记的嵌入向量，使得它们的内积只与它们的相对位置有关。旋转嵌入不需要预先定义或学习位置嵌入向量，而是在网络的每一层动态地添加位置信息。旋转嵌入有一些优点，比如可以处理任意长度的序列，可以提高模型的泛化能力，可以减少计算量，可以适用于线性 Attention 等。  
    (关于 RoPE 的具体细节，可参考[十分钟读懂旋转编码（RoPE） - 知乎](https://zhuanlan.zhihu.com/p/647109286)）  
    

### 2.2 Group Query Attention (GQA)

Llama 2 与 LLaMA 1 的主要架构差异包括上下文长度和分组查询注意力 (GQA) 的增加，如下图：

![](https://pic2.zhimg.com/v2-29e6c1c7a1c67702ccab8270e51f60b5_r.jpg)

**GQA:**

自回归解码的标准做法是使用 KV Cache，即缓存序列中先前 token 的键 (K) 和值 (V) 对，以加快后续 token 的注意力计算。然而，随着上下文窗口或者批量大小的增加，多头注意力（MHA）模型中与 KV Cache 大小相关的内存成本会显著增加。对于大模型，KV Cache 会成为推理时显存应用的一个瓶颈。

对于上述这种情况，有两种主流解决方案：

Multi Query Attention (MQA): 如下图所示，MQA 在所有的 query 之间共享同一份键 (K) 和值 (V) 对

Group Query Attention (GQA): 如下图所示，GQA 在不同的 n (1<=n<=No. of heads) 个 query 之间共享一份键 (K) 和值 (V) 对

![](https://picx.zhimg.com/v2-ccd8357f697d3d75b005a64d3650491f_r.jpg)

虽然仅使用单个键值头的多查询注意力 (MQA) 大大加快了解码器推理。然而，MQA 可能会导致质量下降。而查询注意力 (GQA)，这是一种多查询注意力的泛化，它使用中间（多个，少于查询头的数量）键值头的数量。实验表明，经过预训练的 GQA 实现了接近多头注意力的质量，速度与 MQA 相当。

（关于 MQA，可参考 [https://zhuanlan.zhihu.com/p/634236135](https://zhuanlan.zhihu.com/p/634236135)；

关于 GQA，可参考 [[LLM] Group query attention 加速推理 - 知乎](https://zhuanlan.zhihu.com/p/645865956)；

关于 KV Cache，可参考[大模型推理加速：看图学 KV Cache - 知乎](https://zhuanlan.zhihu.com/p/662498827)）

三、预训练
-----

### 3.1 预训练数据

*   训练语料库包括来自公开可用来源的新混合数据，不包括来自 Meta 产品或服务的数据。努力从已知包含大量关于私人的个人信息的某些站点中删除敏感数据。
*   在 2 万亿个数据上进行训练，因为这提供了良好的性能 - 成本权衡
*   **对大多数事实源数据进行过采样，以增加知识和抑制幻觉**

### 3.2 预训练过程及结果

**3.2.1 预训练超参数**

*   采用 Llama 1 的大部分预训练设置和模型架构
*   优化器：AdamW optimizer， $\beta_{1}$\beta_{1} =0.9， $\beta_{2}$\beta_{2} =0.95
*   学习率：余弦学习率，2000 step 的 warmup，最后 decay 到峰值学习率的 10%
*   weight decay： 0.1
*   gradient clipping：1.0

**3.2.2 预训练的训练损失**

训练 loss 曲线如下所示，**即便训练了 2T 的 token 也暂时没有看到饱和现象**：

![](https://pic1.zhimg.com/v2-674defb332214c4b0a257af8468be0f2_r.jpg)

**3.2.3 Llama 2 预训练模型评估**

*   与开源模型在各个任务上的表现的比较：

![](https://pic3.zhimg.com/v2-631d98016cd987c875cec2ba71f3f500_r.jpg)

除了代码基准测试之外，Llama 2 7B 和 30B 模型在所有类别上都优于相应大小的 MPT 模型。Llama 2 7B 和 34B 在所有类别的基准测试中都优于 Falcon 7B 和 40B。此外，**Llama 2 70B 模型优于所有开源模型。**

*   与闭源模型在各个任务上的表现的比较：

![](https://pic1.zhimg.com/v2-6e4dffead046c83852412844b706566a_r.jpg)

四、Supervised Fine-tuning (SFT)
------------------------------

### 4.1 SFT 数据

为了引导，研究团队从公开可用的指令调优数据开始 SFT 阶段，但后来发现其中许多数据的多样性和质量都不够，特别是在将 LLM 与对话式指令结合起来方面。

*   高质量 SFT 数据收集：使用来自基于供应商的注释的更少但更高质量的示例，SFT 的结果得到了显著改善。
*   Quality> Quantity: 只通过几千个高质量的数据训练的模型效果就优于大规模开源 数据 SFT 训练的模型，这与 Lima 的发现类似：**有限的干净指令调优数据足以达到高水平的质量**。在总共收集了 27,540 个注释后停止标注 SFT 数据。
*   标注偏置的引入：研究团队同时还观察到不同的注释平台和供应商可能导致明显不同的下游模型性能，这突显了即使使用供应商来获取注释时也需要进行数据检查的重要性。为了验证数据质量，研究团队仔细检查了一组 180 个样例，将人工提供的注释与模型生成的样本进行手工审查。令人惊讶的是，我们发现**从结果 SFT 模型中采样的输出往往可以与人类标注者手写的 SFT 数据相竞争**，这表明我们可以重新设置优先级，并将更多的注释工作投入到基于偏好的 RLHF 注释中。

### 4.2 SFT 训练过程及结果

*   余弦学习率，初始学习率 2e-5，weight decay=0.1，batch_size=64，seq_len=4096
*   对于微调过程，每个样本由一个提示（prompt）和一个答案（answer）组成。为了确保模型的序列长度得到正确填充，将训练集中的所有提示和答案连接在一起。使用一个特殊的 token 将提示和答案分隔开。  
    _prompt <sep> answer <eos> prompt <sep> answer_
*   采用自回归目标（autoregressive objective）并将用户提示（prompt）中的标记损失设为零，只用在答案（answer）标记上的 loss 进行反向传播。
*   对模型进行 2 个 epoch 的微调


五、Reinforcement Learning with Human Feedback (RLHF)
---------------------------------------------------

RLHF 是一种模型训练程序，应用于微调的语言模型，以进一步使模型行为与人类偏好和指令遵循保持一致。

（对于 RL 不了解的小伙伴，可以参考这篇[一文看懂什么是强化学习？（基本概念 + 应用场景 + 主流算法）](https://link.zhihu.com/?target=https%3A//easyai.tech/ai-definition/reinforcement-learning/)）

Llama 2 的 RLHF 的主要过程为：人类偏好数据收集 -> 根据数据训练 Reward Model -> 用 RL 的方式迭代式微调（使用 PPO 和 Rejection Sampling）

_（笔者 NOTE: 简而言之，RLHF 就是先通过获取人类偏好的数据来训练一个 reward model，这个 reward model 学习了人类对于语言的偏好；然后这个 model 会用于对 LLM 输出打分，根据所获取的每一步的分数，训练 LLM 往最终产生的整体回复能获取最大分数的方向靠拢）_

### 5.1 人类偏好数据

*   二元比较数据： 主要是因为它允许最大化收集到的提示的多样性
*   标注过程：要求注释者首先编写一个 prompt，然后根据提供的标准在两个抽样的模型的回复（response）之间进行选择；  
    为了使多样性最大化，对给定 prompt 的两个 response 从两个不同的模型生成，并调整温度超参数；  
    除了给参与者一个强制性的选择，还要求标注者标注他们对所选择的回应与备选回应的偏好程度：他们的选择是明显更好、更好、稍微更好、或者几乎一样好 / 不确定
*   标注指标：关注帮助性 (helpfulness，即模型回复满足用户请求并提供请求信息的程度) 和安全性 (safety，即模型回复的安全性)
*   每周分批收集并迭代：在新的 Llama 2-Chat 调优迭代之前，使用最新的 Llama 2-Chat 迭代收集新的偏好数据，再用于后续迭代  
    （这一步骤有助于保持奖励模型分布，并为最新模型保持准确的奖励）

### 5.2 奖励模型 (Reward Model)

奖励模型将模型回复及其相应的提示 (包括来自前一个回合的上下文) 作为输入，并输出一个标量分数来指示模型生成的质量(例如，有用性和安全性)。利用这样的反馈分数作为奖励，我们可以在 RLHF 期间优化 Llama 2-Chat，以更好地调整人类的偏好，提高帮助和安全性。

**5.2.1 奖励模型初始化**

*   两个 Reward Model: 一些研究发现帮助性和安全性有时需要 trade-off，这可能会使单个奖励模型在这两者上同时表现良好具有挑战性。为了解决这个问题，研究团队训练了**两个独立的奖励模型，一个针对有用性（称为_Helpfulness RM_）进行了优化，另一个用于安全（_Safety RM_)**
*   从之前的 Llama 2-chat 检查点初始化：从**预训练的 Llama 2-chat 检查点初始化的奖励模型，这确保两个模型都受益于预训练中获得的知识**。简而言之，奖励模型 “知道”Llama 2-chat 知道什么。这可以防止两个模型会有信息不匹配的情况，这可能导致幻觉
*   RM 模型架构和超参数与预训练的 Llama 2 模型相同，除了用于下一个标记预测的分类头**被用于输出标量奖励的回归头所取代**。

**5.2.2 奖励模型训练目标**

*   研究团队将收集到的成对的人类偏好数据转换为二元排名标签格式 (即选择和拒绝)，并强制选择的响应具有比对应响应更高的分数。
*   **训练 loss：**促使 chosen 的样本得分比 reject 要高，所用的损失函数如下图所示：  
    （其中 x 是 prompt，yc 是标注员选择的模型回复，yr 是标注员拒绝的模型回复）  
    

![](https://pic2.zhimg.com/v2-67cc2eb6714bd0dea47df182a04eee37_r.jpg)

​

*   **加入程度 margin：**为了利用上标注的两条数据的好坏确定程度（明显更好、更好、稍微更好、或者几乎一样好 / 不确定），增加了一个 margin 的 loss 项：

![](https://pic4.zhimg.com/v2-691fb7b84c7f801a1b830fe50fb0ef8b_r.jpg)

*   这个 margin 是一个离散函数，对具有不同响应对使用较大的 margin，对响应相似的对使用较小的 margin，具体值如下：  
    

![](https://pic1.zhimg.com/v2-fe81884dff42497a4c128080a71230a0_r.jpg)

​  

**5.2.3 奖励模型训练数据（混合策略）**

*   **将标记数据与开源偏好数据集混合：**研究团队并没有观察到来自开源偏好数据集的负迁移。因此，将这些开源偏好数据保留在最终的训练 RM 的数据中，因为它们可以为奖励模型提供更好的泛化，并防止奖励幻觉，即 Llama 2-Chat 利用奖励的一些弱点，从而夸大分数
*   **混合策略：**Helpfulness 奖励模型最终是在所有 Meta Helpfulness 数据的基础上训练的，同时还结合了从 Meta Safety 和开源数据集中均匀采样的剩余数据；Safety 奖励模型则是在所有 Meta Safety 和 Anthropic Harmless 数据上进行训练的，同时还混合了 Meta Helpfulness 和开源的 Helpfulness 数据，比例为 90/10 （在 10%Helpfulness 数据的设置下，对于那些被所选和被拒绝的回答都是安全的的准确性尤为有益）

**5.2.4 奖励模型训练**

*   训练一个 epoch （训练长了会导致过拟合）
*   使用和基础模型相同的训练超参数
*   70B 的模型使用 5e-6 的学习率，其他的使用 1e-5 学习率
*   余弦学习旅衰减，学习率降低到最大学习率的 10%；使用总步数的 3% 作为 warmup
*   每个 batch 有 512 pairs 数据
*   实现的效果：

![](https://pic3.zhimg.com/v2-ba5bef62c7de3b374670a101275c3e8a_r.jpg)

*   奖励模型的准确性是 Llama 2-Chat 最终性能的最重要代表之一。虽然综合评估生成模型的最佳实践是一个开放的研究问题，但奖励的排名任务没有任何歧义。因此，在其他条件相同的情况下，奖励模式的改进可以直接转化为 Llama 2-Chat 的改进。

### 5.3 RLHF 迭代式微调

**5.3.1 RLHF 对应的基本概念**

*   **Agent（智能体）：**强化学习的本体；在此情境下为经过 SFT 后的 Llama2-Chat
*   **Environment （环境）：**智能体外的一切，由状态集合组成；在此情境下为与人对话着整个场景
*   **State（状态）：**某个时刻环境的状态；在此情境下为用户输入的 prompt 或者 prompt 加上输出的回复  
    （Note：对于 answer 的第二个词，可以把 prompt+answer 的第一个词当作新的 state，而不只是把 prompt 当作 state，状态转移蕴含在 transformer 内部）
*   **Action（动作）：**智能体根据环境状态做出的动作；在此情境下为对应于 prompt 而输出的回复（answer）
*   **Reward（奖励）：**智能体在执行一个动作后，获得的反馈
*   **整个流程：**用户初始输入 prompt 是第一个 state，只输入一次，然后模型输出一串 action（回答的单词），得到一个 reward，模型并没有在每个 action 之后得到新的 state，二是在获取 answer 的第二个词后，把 prompt+answer 的第一个词当作新的 state，而不只是把 prompt 当作 state，状态转移蕴含在 transformer 内部

**5.3.2 RLHF 整体流程**

当训练完奖励模型，能够获取对英语 LLM 的输出的奖励之后，为了模型不对某些提示的特定分布过拟合，研究团队使用了迭代式微调的策略，为 RLHF 模型训练了连续版本，这里称为 RLHF-V1， ...， RLHF-V5 等。  

研究团队尝试了两种 RLHF 微调算法：

*   **Proximal Policy Optimization （PPO）：**PPO 算法是一种强化学习中更新策略（Policy）的算法；相较于传统的 Policy Gradient 算法，Policy Gradient 算法对步长十分敏感，但是又难以选择合适的步长，在训练过程中新旧策略的的变化差异如果过大则不利于学习；PPO 提出了新的目标函数可以再在个训练步骤实现小批量的更新，解决了 Policy Gradient 算法中步长难以确定的问题。  
    （关于 PPO 的具体内容，可参考李宏毅老师的视频（讲的超级清楚）[(选修)To Learn More - Proximal Policy Optimization (PPO)_哔哩哔哩_bilibili](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Wv411h7kN%3Fp%3D124%26vd_source%3D8953fe49a44d66291656941b4278f257)）  
    
*   **Rejection Sampling（拒绝采样）：**一种对于复杂问题的采样策略；在强化学习中，智能体（agent）需要从环境中获得高质量的训练数据。使用 Rejection Sampling，可以更有效地从环境中采样出对学习策略更有价值的情况，从而提高训练数据的质量。  
    （关于拒绝采样，可参考[蒙特卡洛采样之拒绝采样（Reject Sampling） | TwistedW's Home](https://link.zhihu.com/?target=http%3A//www.twistedwg.com/2018/05/30/MC-reject-sampling.html)）  
    - 研究团队在应用拒绝采样时，从模型中采样 K 个输出，并根据奖励模型计算出的奖励选择最佳候选（与 Constitutional AI: Harmlessness from AI Feedback 论文方法一致）。研究团队进一步使用选择的输出进行梯度更新。对于每个提示，获得最高奖励分数的样本被认为是新的黄金标准，随后在新的排序样本集上微调模型，强化奖励  
    
*   **两种 RL 算法的主要区别在于：**

1.  **深度：**PPO 中，在步骤 t 训练期间，样本是在上一步梯度更新后从 t-1 更新的模型策略的函数。在拒绝采样中，在应用类似 SFT 的微调之前，在给定之前模型的初始策略的情况下对所有输出进行采样以收集新数据集。然而，由于应用了迭代模型更新，两种 RL 算法之间的根本差异不太明显
2.  **广度：**在拒绝采样中，该模型为给定提示探索 K 个样本，而 PPO 只进行了一个样本的探索  
    （使用拒绝采样可以更有效地从环境中采样出对学习策略更有价值的情况；用拒绝采样获取了高质量的数据后，PPO 可以用来优化策略。由于 PPO 通过限制策略更新幅度来减少训练过程中的性能波动，它可以更有效地利用这些高质量数据来进行稳定的策略学习。结合这两种方法还可以帮助在探索（尝试新的行为以获取更多信息）和利用（使用已知信息来获得最佳结果）之间找到更好的平衡。Rejection Sampling 可以用来探索更多样的情境，而 PPO 则可以确保这些探索得到有效利用。）  
    

*   **结合拒绝采样和 PPO：**在 RLHF (V4) 之前，只使用拒绝采样微调，之后，研究团队将两者结合起来，在再次采样之前在 Rejection Sampling checkpoint 上应用 PPO。

**5.3.3 拒绝采样（Rejection Sampling）**  

*   **用 70B Llama 2-Chat 执行拒绝采样：**仅使用最大的 70B Llama 2-Chat 执行拒绝采样。所有较小的模型都根据来自较大模型的拒绝采样数据进行微调，从而将大模型能力提炼为较小的模型（相当于模型蒸馏）  
    
*   **使用所有 RLHF 迭代模型进行拒绝采样：**在每个迭代阶段，研究团队从模型中为每个 prompt 采样 K 个答案。然后，使用当时实验可访问的最佳奖励模型对每个样本进行评分，并选择给定 prompt 的最佳答案。研究团将所有迭代中（RLHF-V1、RLHF-V2 、RLHF-V3）表现最好的样本都纳入训练数据。  
    
*   **拒绝抽样的增益：**如下图，最大曲线和中位数曲线之间的差异可以被解释为在最佳输出上进行微调的潜在收益。正如预期的那样，随着样本数量的增加，这种差异增大（即更多样本，更多机会生成良好的轨迹），而中位数保持不变。在样本中，探索和获得最大奖励之间存在直接联系。温度参数对于探索也起着重要作用，因为较高的温度能够使采样更多样化的输出。  
    

![](https://pica.zhimg.com/v2-2c4e016fbf245200537627cd309e39a4_r.jpg)

*   **经过拒绝采样训练后的最大奖励曲线：**在下图中展示了 Llama 2-Chat-SFT（左图）和 Llama 2-Chat-RLHF（右图）的最大奖励曲线，这些曲线是在不同温度下进行 N 次样本采样（其中 N ∈ [1, . . . , 100]）得到的。可以观察到，在迭代模型更新的过程中，平均 reward 在增加；同时，最佳温度是不固定的，且 RLHF 对温度进行了直接影响。

![](https://pica.zhimg.com/v2-a17c607576f84fdba38f9a8843cae268_r.jpg)

​

**5.3.4 PPO**

PPO 是一种 off-policy 的手段，使用奖励模型作为真实奖励函数 (人类偏好) 的估计，使用预训练的语言模型作为 policy 进行优化。  

*   **优化目标：**优化目标就是提升 reward，同时与原始模型的输出加个 KL 散度约束（为了训练稳定性，并且缓解 reward hacking 情况）  
    

![](https://pic3.zhimg.com/v2-e7b32dde73992f8e4fabe73576f68036_r.jpg)

*   其中 $R_{c}$R_{c} 是安全性奖励和帮助性奖励的分段组合。其计算公式如下：  
    

![](https://pic2.zhimg.com/v2-ce0eae48aaab9bbce9adffd80d485b91_r.jpg)

​  
将最终的线性分数进行白化，以增加稳定性并与上面的 KL 惩罚项正确平衡：  

![](https://pic1.zhimg.com/v2-0c52c650bcf340ce5926b59fdae248a0_1440w.jpg)

​  

**5.3.5 训练细节**

*   AdamW： $\beta_{1}$\beta_{1} = 0.9, $\beta_{2}$\beta_{2} = 0.95
*   Weight decay=0.1，Gradient clipping=1.0
*   Constant learning rate=10−6
*   Batch Size：512
*   PPO clip threshold=0.2, mini batch size=64, take one gradient step per mini-batch
*   KL 惩罚系数：7B and 13B 采用 0.01，34B 和 70B 采用 0.005  
    

六、多轮一致性的系统消息
------------

### 6.1 模型忘记初始指令

在对话设置中，有些指令应该适用于所有对话回合，例如要简洁回复，或者 “扮演” 某个公众人物。当向 Llama 2-Chat 提供这样的指令时，后续的回复应始终遵守这些限制。  

然而，最初的 RLHF 模型在对话进行几个回合后往往会忘记初始指令，如下图（左图）所示。为了解决这些问题，研究团队提出了 Ghost Attention（GAtt）方法。GAtt 使得对话在多个回合内能够保持控制，如下图（右图）所示：  

![](https://pic1.zhimg.com/v2-b11b7d220d6d056ac62a743238f310bc_r.jpg)

​

### 6.2 Ghost Attention (GAtt)

这是一个受 Context Distillation 启发的非常简单的方法，通过对微调数据进行干预来帮助注意力在多阶段的过程中聚焦。GAtt 使得对话在多个回合内能够保持控制。  

**具体流程：**  

*   假设可以访问两个人之间的多轮对话数据集（例如，用户和助手之间的对话），其中包含一系列消息 [u1, a1, ..., un, an]，其中 un 和 an 分别对应第 n 轮对话的用户和助手消息。然后，研究团队定义一个指令（inst），在整个对话过程中应该被遵守。例如，指令可以是 "扮演" 某个角色。然后，将这个指令合成地连接到对话中所有的用户消息上。
*   接下来，可以使用最新的 RLHF 模型从这个合成数据中进行采样。现在有了一个上下文对话和用于微调模型的样本，这个过程类似于拒绝抽样。然而，研究团队并不是在所有上下文对话回合中都加入指令，而是只在第一个回合中加入，这样会导致一个训练时的问题，即系统消息（即最后一轮之前的所有中间助手消息）与原来的样本不匹配。为了解决这个问题，以免影响训练，研究团队简单地将之前回合中的所有标记的损失设置为 0，包括助手消息。
*   对于训练指令，研究团队创建了一些合成的限制供采样，例如兴趣爱好（"您喜欢（），例如网球"），语言（"说 ()，例如法语"），或者公众人物（"扮演 ()，例如拿破仑"）。为了获得兴趣爱好和公众人物的列表，研究团队让 Llama 2-Chat 来生成，避免了指令与模型知识不匹配的问题（例如，让模型扮演它在训练中没有遇到过的角色）。为了使指令更加复杂和多样化，研究团队通过随机组合上述限制来构造最终的指令。在构造用于训练数据的最终系统消息时，研究团队还会将一半的原始指令修改为更简洁的形式，例如 "Always act as Napoleon from now" 会变为 "Figure: Napoleon"。这些步骤生成了一个 SFT 数据集，用于微调 Llama 2-Chat。  
    

GAtt 评测：为了说明 GAtt 如何帮助在微调期间重塑注意力，在下图中展示了模型的最大注意力激活（与没有 GAtt 的模型 (左) 相比，配备 GAtt 的模型 (右) 在更大的对话部分中保持了与系统消息相关的大量注意力）：  

![](https://picx.zhimg.com/v2-e71d2931737bdeeb01b3dc0557a4ee6f_r.jpg)

​

七、实验结果
------

### 7.1 基于模型的评估结果

*   为了测评不同的 SFT 和 RLHF 版本在安全性和有用性两个方面的进展情况，进行了内部的安全性和有用性奖励模型进行度量。在这组评估中，在 RLHF-V3 版本之后在两个方面都优于 ChatGPT（一种基线模型），即无害性（harmlessness）和有用性（helpfulness）均高于 50%。  
    
*   为了公平比较，额外使用 GPT-4 进行最终结果的计算，以评估哪个生成模型更受青睐。为避免任何偏见，ChatGPT 和 Llama 2-Chat 输出在 GPT-4 提示中的顺序会被随机交换。如预期，Llama 2-Chat 相对于 ChatGPT 的胜率变得不太显著，不过最新的 Llama 2-Chat 仍超过 60% 的胜率。  
    

![](https://pica.zhimg.com/v2-0cd66cfa5679e458224ef2844851e6a0_r.jpg)

### ​  
7.2 基于人工的评测结果

![](https://pica.zhimg.com/v2-95c854359c586b2f4218b21644649b28_r.jpg)

笔者 NOTE：由于篇幅原因，本篇详细地介绍了 RLHF 的内容，简短地带过了 Llama RLHF 实现的结果。Llama 原文中对大模型研发的每个环境都讲解地非常详细，对于其他内容（模型、数据的安全性，GQA 的优势、训练数据的分析以及其他开放问题的探讨），有兴趣的小伙伴可以去看下原文，或者参考 [Llama2 详解：开源世界最优秀的大语言模型 - 知乎](https://zhuanlan.zhihu.com/p/645381497)  

我们下期见～