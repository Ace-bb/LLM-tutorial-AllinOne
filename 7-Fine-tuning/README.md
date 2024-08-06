

What is fine-tuning?
---------------------------------
机器学习中的微调是针对特定任务或用例调整预训练模型的过程。它已成为一种基本的深度学习技术，特别是在用于生成人工智能的基础模型的训练过程中。

## 背景、动机
Fine-tuning背后的动机是，从本质上讲，提高一个已经学习到了大量与当前任务相关知识的预训练模型的能力，比从头开始训练一个新模型更容易、更便宜。特定是对于具有数百万甚至数十亿参数的深度学习模型尤其如此，例如在自然语言处理（NLP）领域或复杂的卷积神经网络（CNN）和视觉领域日益突出的大型语言模型（LLM） Transformer (ViT) 用于计算机视觉任务，例如图像分类、对象检测或图像分割任务等。

通过迁移学习来激发预训练模型的能力，微调可以减少训练一个针对利特定任务和业务需求的大型模型所需的昂贵的计算成本和标记数据量。例如，微调可以用于简单地调整预训练的LLM的对话语气或预训练的图像生成模型的插图风格；它还可以用来用专有数据或专门的、特定领域的知识来补充模型原始训练数据集中的知识。

例如，一个通用大模型涵盖了许多语言信息，并能够进行流畅的对话。但如果涉及到某个专业领域，例如在医药方面能够很好地回答患者的问题，**就需要为这个通用大模型提供更多新的优质数据供其学习和理解**。为了确定模型能够正确回答，我们就需要对基础模型进行微调。

Fine-tuning vs. training
---------------------------------
### (Pre-)Training

在训练开始时（或者在这种情况下为预训练），模型尚未“学习”任何内容。训练从模型参数的随机初始化开始——应用于神经网络中每个节点的数学运算的不同权重和偏差。

训练分两个阶段迭代进行：
- 在前向传播中，模型对训练数据集中的一批样本输入进行预测，损失函数计算模型对每个输入的预测值与“正确答案”之间的差异（或损失、或基本事实）；
- 在反向传播期间，优化算法（通常是梯度下降）用于调整整个网络的模型权重以减少损失。对模型权重的这些调整就是模型“学习”的方式。该过程在多个训练时期重复，直到模型被认为经过充分训练。

传统的监督学习通常用于为图像分类、对象检测或图像分割等计算机视觉任务预训练模型，它使用标记数据：labels (or annotations)提供可能答案的范围以及每个答案的真实输出样本。

LLM 通常通过自监督学习 (self-supervised learning SSL) 进行预训练，其中模型通过预训练文本任务进行学习，这些任务旨在从固定结构的未标记数据中挖掘真实答案。这些预训练文本任务(pretext task)包含大量对下游任务有用的知识。他们通常采用以下两种方法之一：

- Self-prediction：屏蔽原始输入的某些部分并让模型预测屏蔽的部分。这是LLM的主要训练方式。就是mask learning

- 对比学习：训练模型学习相关输入的相似嵌入和不相关输入的不同嵌入。这主要用于为少样本或零样本学习而设计的计算机视觉模型，例如对比语言图像预训练（CLIP）。

因此，自监督学习允许在训练中使用海量数据集，而无需标注数百万或数十亿个数据。这节省了大量的人力成本，但仍然需要大量的计算资源。

### Fine-tuning
相反，微调是进一步训练已完成预训练的模型的技术。使用预训练模型的先前知识作为起点，通过在较小的特定于任务的数据集上进行训练来微调模型。

虽然该特定于任务的数据集理论上可以用于初始训练，但在小数据集上从头开始训练大型模型存在过度拟合的风险：模型可能会在训练示例上表现良好，但对新数据的泛化能力较差。这将使模型不适合其给定任务并违背模型训练的目的

因此，微调提供了两全其美的方法：利用从大量数据预训练中获得的广泛知识和稳定性，并训练模型对更详细、更具体概念的理解。鉴于开源基础模型的实力不断增强，通常无需任何预训练的财务、计算或后勤负担即可享受到这些好处。

How does fine-tuning work?
---------------------------------

微调使用预训练模型的权重作为起点，在较小的示例数据集上进行进一步训练，这些示例更直接地反映了模型将用于的特定任务和用例。它通常需要监督学习，但也可能涉及强化学习、自我监督学习或半监督学习。

用于微调的数据集传达了预训练模型正在微调的特定领域知识、风格、任务或用例。例如：

*   为通用语言预训练的 LLM 可能会进行微调，以便使用包含相关编程请求和每个请求的相应代码片段的新数据集进行编码。
    
*   用于识别某些鸟类的图像分类模型可以通过额外的标记训练样本来学习新物种。
    
*   法学硕士可以通过对代表该风格的样本文本进行自我监督学习来学习模仿特定的写作风格。
    

[_半监督学习_](https://www.ibm.com/topics/semi-supervised-learning)是机器学习的一个子集，它结合了标记和未标记的数据，当场景需要监督学习但合适的标记示例稀缺时，半监督学习是有利的。半监督微调在计算机视觉 1 和 NLP2 任务中都产生了有希望的结果，并有助于减轻获取足够数量的标记数据的负担。

微调可用于更新整个网络的权重，但出于实际原因，情况并非总是如此。存在各种各样的替代微调方法，通常在_参数高效微调_ （PEFT） 的总称下称为微调方法，这些方法仅更新模型参数的选定子集。PEFT 方法可以减少计算需求，减少[灾难性遗忘](https://research.ibm.com/publications/forget-me-not-reducing-catastrophic-forgetting-for-domain-adaptation-in-reading-comprehension)（一种微调导致模型核心知识丢失或不稳定的现象），通常不会对性能做出有意义的妥协。

鉴于微调技术的种类繁多以及每种技术固有的许多变量，要实现理想的模型性能，通常需要多次迭代训练策略和设置，调整数据集和超参数，如批量大小、学习率和正则化项，直到达到令人满意的结果（根据与您的用例最相关的指标）。

全面微调

从概念上讲，最直接的微调方法是简单地更新整个神经网络。这种简单的方法本质上类似于预训练过程：完全微调和预训练过程之间唯一的根本区别是正在使用的数据集和模型参数的初始状态。

为了避免微调过程中的不稳定变化，某些_超参数_（影响学习过程但本身不是可学习参数的模型属性）可能会在预训练期间相对于其规格进行调整：例如，较小的_学习率_（这会降低每次更新模型权重的幅度）不太可能导致灾难性的遗忘。

参数高效微调 （PEFT）

完全微调，就像它类似于预训练过程一样，对计算要求非常高。对于具有数亿甚至数十亿个参数的现代深度学习模型，它通常成本高昂且不切实际。

参数高效微调 （PEFT） 包含一系列方法，用于减少需要更新的可训练参数数量，以便有效地将大型预训练模型适应特定的下游应用程序。在此过程中，PEFT 大大减少了产生有效微调模型所需的计算资源和内存存储。PEFT 方法通常被证明比完全微调方法更稳定，特别是对于 NLP 用例。3  
 

**部分微调  
**部分微调方法也称为_选择性微调_，旨在通过仅更新对相关下游任务的模型性能最关键的预训练参数的选定子集来减少计算需求。其余参数被 “冻结”，确保它们不会被更改。

最直观的部分微调方法是仅更新神经网络的外层。在大多数模型架构中，模型的内层（最靠近输入层）仅捕获广泛的通用特征：例如，在用于图像分类的 CNN 中，早期层通常能够识别边缘和纹理; 随后的每一层都会逐渐识别出更精细的特征，直到在最外层预测出最终分类。一般来说，新任务（模型正在微调的任务）与原始任务越相似，内层的预训练权重对于这个新的相关任务就越有用，因此需要更新的层就越少。

其他部分微调方法包括仅更新模型的层范围偏差项（而不是特定于节点的权重）4 和 “稀疏” 微调方法，这些方法仅更新整个模型中总体权重的选定子集。5

  
**增材制造微调  
**加法方法不是微调预训练模型的现有参数，而是向模型添加额外的参数或层，冻结现有的预训练权重，并仅训练那些新组件。这种方法通过确保原始预训练权重保持不变来帮助保持模型的稳定性。

虽然这可能会增加训练时间，但它可以显着降低内存需求，因为要存储的梯度和优化状态要少得多：根据 Lialin 等人的说法，训练模型的所有参数需要的 GPU 内存是单独模型权重的 12-20 倍。6 通过_量化_冻结模型权重，可以进一步节省内存：降低用于表示模型参数的精度，在概念上类似于降低音频文件的比特率。

加法方法的一个子分支是[_提示调优_](https://research.ibm.com/blog/what-is-ai-prompt-tuning)。从概念上讲，它类似于[提示工程](https://www.ibm.com/topics/prompt-engineering)，它指的是定制 “硬提示”（即人类用自然语言编写的提示），以引导模型达到所需的输出，例如通过指定特定的语气或提供促进[小样本学习](https://www.ibm.com/topics/few-shot-learning)的示例。提示调优引入了 AI 编写的_软提示_：可学习的向量嵌入，这些嵌入连接到用户的硬提示。提示调整不是重新训练模型，而是需要冻结模型权重，而是训练软提示本身。快速高效、及时的调优使模型能够更轻松地在特定任务之间切换，尽管在[可解释性](https://www.ibm.com/topics/explainable-ai)方面有所权衡。  

**适配器  
**加法微调的另一个子集注入_适配器模块_（添加到神经网络中的新任务特定层）并训练这些适配器模块，而不是微调任何预训练的模型权重（这些权重是冻结的）。根据在 BERT 掩码语言模型上测量结果的原始论文，适配器获得了与完全微调相当的性能，而训练的参数数量仅为 3.6%。7

  
**重参数化**  
基于重参数化的方法，如_低秩自适应 （LoRA），_利用高维矩阵的低秩变换（如转换器模型中预训练模型权重的大量矩阵）。这些低秩表示省略了无关紧要的高维信息，以便捕获模型权重的底层低维结构，从而大大减少了可训练参数的数量。这大大加快了微调速度，并减少了存储模型更新所需的内存。

LoRA 避免直接优化模型权重矩阵，而是优化模型权重的更新矩阵 （或_增量权重_）， 该矩阵入到模型中. 反过来，权重更新矩阵又表示为两个较小的（即_较低秩_）矩阵，大大减少了要更新的参数数量，这反过来又大大加快了微调速度并减少了存储模型更新所需的内存。预训练的模型权重本身保持冻结状态。

LoRA 的另一个好处是， 由于被优化和存储的不是新的模型权重，而是原始预训练权重和微调权重之间的差异 （或增量），因此可以根据需要 “交换” 不同的特定任务 LoRA，以使预训练模型（其实际参数保持不变）适应给定的用例。

已经开发了多种 LoRA 衍生产品， 例如 _QLoRA，_ 它通过在 LoRA 之前量化变压器模型来进一步降低计算复杂性.


Fine-tuning large language models
---------------------------------

微调是 LLM 开发周期的重要组成部分，它允许基础模型的原始语言能力适应各种用例，从[聊天机器人](https://ibm.com/topics/chatbots)到编码，再到其他创意和技术领域。

LLM 使用自监督学习对大量未标记数据进行预训练。自回归语言模型，如 OpenAI 的 GPT、Google 的 Gemini 或 Meta 的 [Llama 模型](https://www.ibm.com/topics/llama-2)，经过训练可以简单地预测序列中的下一个单词，直到它完成。在预训练中，为模型提供从训练数据中提取的样本句子的开头，并重复执行任务，预测序列中的下一个单词，直到样本结束。对于每个预测，原始样本句子的实际下一个单词用作基本事实。

**虽然这种预训练产生了强大的文本生成功能，但它并不能实际理解用户的意图。从根本上讲，自回归 LLM 实际上并不响应提示; 他们只_向其附加文本_。**如果没有提示工程形式的非常具体的指导，预先训练的 LLM（尚未进行微调）只是以语法连贯的方式预测由提示启动的给定序列中的下一个单词可能是什么。如果系统提示 “_教我如何制作简历_”，LLM 可能会回答 “_使用 Microsoft Word_”。这是完成句子的有效方式，但与用户的目标不一致。该模型可能已经从其预训练语料库中包含的相关内容中收集了大量的简历写作知识，但如果不进行微调，可能无法访问这些知识。

因此，微调过程不仅在为您或您的企业的独特基调和用例定制基础模型方面发挥着至关重要的作用，而且在使它们完全适合实际使用方面也起着至关重要的作用。

指令调优

_指令调优_是监督微调 （SFT） 的一个子集，通常用于微调 LLM 以供聊天机器人使用，它使 LLM 生成更直接满足用户需求的响应：换句话说，更好地遵循指令。按照格式（_prompt、response_）标记的示例（其中提示示例包含面向教学的任务，例如 “_将以下句子从英语翻译成西班牙语_” 或 “_将以下句子分类为肯定或否定_”——演示如何响应代表各种用例的提示，如问答、摘要或翻译。在更新模型权重以最小化模型输出和标记样本之间的损失时，LLM 学会了以更有用的方式将文本附加到提示中，并更好地遵循一般的说明。

继续前面的提示示例 “_教我如何写简历_”，用于 SFT 的数据集可能包含许多（_提示、响应_）对，这表明响应以 “_教我如何_” 开头的提示的理想方式是提供逐步的建议，而不仅仅是完成句子。

来自人类反馈的强化学习 （RLHF）

虽然指令调优可以教给模型有形的、直接的行为，比如如何构建其反应，但通过标记的例子来教授抽象的人类品质，如乐于助人、事实准确性、幽默感或同理心，可能会非常费力和困难。

为了更好地将模型输出与理想的人类行为保持一致，特别是对于聊天机器人等对话用例，SFT 可以辅以强化学习，更具体地说，[是来自人类反馈的强化学习 （RLHF）。](https://www.ibm.com/topics/rlhf)RLHF，也称为_从人类偏好中汲取的强化学习_，有助于通过离散示例微调复杂、定义不明确或难以指定的品质模型。

以喜剧为例：要用 SFT 教模型 “有趣”，不仅需要编写（或获取）足够多的笑话来构成可学习模式的成本和劳动，而且还需要给定的数据科学家认为有趣的东西与用户群会觉得有趣的东西保持一致。RLHF 本质上提供了一种数学上众包的替代方案：促使 LLM 产生笑话，并让人类测试人员评估其质量。这些评级可以用来训练_一个奖励模型_，以预测将获得积极反馈的笑话类型，反过来，这个奖励模型可以用来通过强化学习来训练 LLM。

更实际地说，RLHF 旨在解决 LLM 的生存挑战，例如[幻觉](https://www.ibm.com/topics/ai-hallucinations)、反映训练数据中固有的社会偏见或处理粗鲁或对抗性的用户输入。
