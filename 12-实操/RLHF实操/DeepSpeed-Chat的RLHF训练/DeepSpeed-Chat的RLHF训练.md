
# DeepSpeed-Chat的RLHF训练
注意：官方代码为：[https://github.com/microsoft/DeepSpeedExamples/tree/ds-chat-3/applications/DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/ds-chat-3/applications/DeepSpeed-Chat)

分支为：`ds-chat-3`

本教程使用的是该分支的代码。


DeepSpeed-Chat 采用了 InstructGPT 的训练方法，整合了一个完整的端到端训练流程（如图1所示）。这个流程主要分为以下三步：

1. **监督微调（SFT）**：使用人类提供的精选回答对模型进行微调，得到一个初步的 SFT 模型。
2. **奖励模型训练**：通过人类对同一问题的多个答案进行打分的数据，训练一个独立的奖励模型。
3. **强化学习微调**：根据奖励模型的反馈，利用 PPO 算法对 SFT 模型进行进一步的微调。

这个流程通过逐步优化模型，使其能够更好地理解和生成符合人类期望的回答。

在经典的三步训练过程中，DeepSpeed-Chat 引入了两项特色功能来优化模型性能，分别是**指数移动平均（EMA）**和**混合训练**。

- **EMA**：这项技术在 InstructGPT 的研究中被证明，相比传统方法，它能更有效地提升模型的响应质量。
- **混合训练**：它的目的是将预训练目标（即预测下一个词）与 PPO 目标结合起来。

虽然这些优化技巧在很多开源框架中并没有得到足够的重视（可能是因为它们对训练流程的影响不太明显），但为了充分复现 InstructGPT 的训练方法，并获得更好的模型表现，这些功能是必不可少的。

![](https://gitee.com/Ace_bb/static_resource_cloud/raw/master/LLMBook/LLMBOOK1/images/bf05c6fd262795f5b264c32cf2d5b34aa0eca5fac5db36740bee3f8b203c8d2e.jpg)
图1 DeepSpeed-Chat 的端到端训练流程图


`train.py` 示例脚本提供了完整的三步训练流程，并且支持通过多个命令行参数进行配置，比如：
- 模型类型
- 模型大小
- 显卡数量等

为了满足不同用户的需求，DeepSpeed-Chat 提供了更灵活的可配置性：
- 如果用户只需要在第一步或第二步微调预训练模型，可以直接选择对应的步骤。
- 如果用户已经有自己的 `Actor` 和 `Reward` 模型的检查点，可以直接执行 RLHF 流程的第三步。

这种设计让用户能够根据自己的需求选择不同的训练方式，而不必强制完成所有训练步骤。


### 6.3.1 数据收集与整理

DeepSpeed-Chat 支持使用多个不同来源的数据集来训练模型，从而提升模型的质量。它主要提供了以下两个功能：

- **抽象数据功能**：DeepSpeed-Chat 提供了一个抽象数据集层，能够将不同数据集的格式统一起来。这样一来，使用不同来源的数据集变得更加方便和灵活。
- **数据拆分/混合功能**：DeepSpeed-Chat 还支持数据拆分和混合功能，允许在模型训练的三个步骤中，将多个数据集进行合理的拆分和混合。这种方式可以更好地利用不同数据集的信息，从而提升训练效果和模型性能。

除了使用示例脚本中的数据集，我们还可以添加和使用自己的数据集。具体步骤如下：

1. **创建数据文件夹**  
   在 `DeepSpeed-Chat` 目录下新建一个名为 `data` 的文件夹，用于存放所有的训练数据和评估数据。

2. **准备数据文件**  
   在 `data` 文件夹下创建两个文件：
   - `train.jsonl`：用于训练数据
   - `eval.jsonl`：用于评估数据

   文件中的数据格式要求如下：
   - 每个文件的内容是一个 JSON 列表。
   - 列表中的每一项是一个字典，格式为：
     json
     {"prompt": "Human: 问题 Assistant:", "chosen": "正确回答", "rejected": "错误回答"}
     

3. **修改代码以支持数据更新**  
   由于我们可能会频繁修改自定义数据集的内容，因此需要确保每次修改后数据能够及时刷新。具体操作如下：
   - 打开 `training/utils/data/data_utils.py` 文件。
   - 找到 `create_prompt_dataset` 函数。
   - 将该函数的 `reload` 参数设置为 `True`。如果不设置，修改后的数据将不会刷新缓存文件。

通过以上步骤，你就可以轻松使用自己的数据集进行训练和评估了。

在RLHF（基于人类反馈的强化学习）训练中，数据集中最复杂的就是奖励模型的训练数据。这类数据不仅需要包含好的回答，还需要有坏的回答作为对比。目前，中文的奖励模型训练数据非常稀缺，大部分开源的数据都是英文的。

Anthropic公司在一篇论文《Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback》（https://arxiv.org/abs/2204.05862）中提到了一个叫`hh-rlhf`的数据集（https://huggingface.co/datasets/Dahoas/full-hh-rlhf）。这个数据集已经被广泛用于像`Llama 2`这样的模型中，为后续的强化学习训练奖励模型提供了重要支持。

此外，这个数据集还被翻译成了中文版本`dikw/hh_rlhf_cn`（https://huggingface.co/datasets/dikw/hh_rlhf_cn）。在本章中，我们将基于这个中文数据集进行`DeepSpeed-Chat`的实践。

我们需要按照 `DeepSpeed-Chat` 规定的处理格式来集成 `dikw/hh_rlhf_cn` 数据集。具体步骤如下：

1. 在 `training/utils/data/raw_datasets.py` 文件中添加一个新的类，用于定义数据格式。
2. 确保这个新类遵循 `PromptRawDataset` 类中定义的 API 和格式，以保证数据格式的一致性。

通过以上步骤，我们可以确保数据集的处理符合 `DeepSpeed-Chat` 的要求。

```python
# Chinese dataset
class DikwHhRlhfCnDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "dikw/hh_rlhf_cn"
        self.dataset_name_clean = "dikw_hh_rlhf_cn"
    def get_train_data(self):
        return self.raw_datasets["train"]
    def get_eval_data(self):
        return self.raw_datasets["test"]
    def get_prompt(self, sample):
        context = ""
        for item in sample["context"]:
            context += f"{item['role']}:{item['text']}|"
        context += "assistant:"
        return context
    def get_chosen(self, sample):
        return sample["chosen"]["text"]
    def get_rejected(self, sample):
        return sample["rejected"]["text"]
    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + sample['chosen']["text"]
    def get_prompt_and_rejected(self, sample):
        return self.get_prompt(sample) + sample['rejected']["text"]
```

1. 首先，打开文件 `training/utils/data/data_utils.py`，找到里面的 `get_raw_dataset` 函数。
2. 然后，在这个函数里添加一个新的 `if` 条件语句。
3. 这个 `if` 语句的作用是判断当前处理的数据集名称是否匹配你提供的新数据集名称。
4. 注意，`dataset_name` 这个字符串需要和你在训练脚本中指定的数据集名称保持一致。

简单来说，就是：
- 修改 `get_raw_dataset` 函数。
- 添加一个 `if` 条件，检查 `dataset_name` 是否是你想要的新数据集名称。
- 确保 `dataset_name` 和训练脚本中用的名称一致。

```python
def get_raw_dataset(dataset_name, output_path, seed, local_rank):
    if "Dahoas/rm-static" in dataset_name:
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                       local_rank, dataset_name)
    elif "Dahoas/full-hh-rlhf" in dataset_name:
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                       local_rank, dataset_name)
    elif "dikw/hh_rlhf_cn" in dataset_name:
        retu rn  raw_datasets.DikwHhRlhfCnDataset(output_path,  seed,  local_rank, 
dataset_name)
```

最后，在训练脚本的 `--data_path` 参数里加上新数据集的 `dataset_name`。需要注意的是，三步训练的脚本文件里都得做相应的修改。

```python
# tr aining/step1_supervised_finetuning/training_scripts/opt/single_node/run_1.3b.sh deepspeed main.py --data_path dikw/hh_rlhf_cn \
# tr aining/step2_reward_model_finetuning/training_scripts/opt/single_node/run_350m.sh
deepspeed main.py --data_path dikw/hh_rlhf_cn \
# training/step3_rlhf_finetuning/training_scripts/opt/single_node/run_1.3b.sh
deepspeed --master_port 12346 main.py --data_path dikw/hh_rlhf_cn \
```

当然，如果你已经有一个本地的数据集，或者你手动从 Hugging Face 下载了数据集，你也可以把本地路径添加到 `--data_path` 参数里。比如：

- 使用相对路径：`--data_path ./relative/dataset_dir/dataset`
- 使用绝对路径：`--data_path /absolute/dataset_dir/dataset`

需要注意的是，尽量不要在本地路径里加 `data/`，否则可能会导致加载数据集时出问题。

另外，有些数据集只在第一步监督微调（SFT）的时候才会用到。这种情况下，你应该把 `dataset_name` 添加到 `--sft_only_data_path` 参数里，而不是 `--data_path` 参数。


这里有几个关键点需要注意：

1. 如果我们**只进行第一步的监督微调**，而不做第二步的奖励模型微调和第三步的强化学习训练，那么在 `--sft_only_data_path` 参数中添加多个数据集是有好处的。

2. 但如果我们**想完成完整的三步训练**（包括监督微调、奖励模型微调和强化学习训练），那么在 `--sft_only_data_path` 参数中添加太多数据集可能会带来负面影响。这是因为：
   - 这些数据集可能与第二步和第三步使用的数据集**分布不一致**。
   - 这种分布差异可能导致训练过程不稳定，最终影响模型的效果。


### 6.3.2 有监督微调

通过以下几条命令就可以使用标注好的数据对预训练模型进行有监督微调。

```bash
python train.py --step 1 --deployment-type single_gpu   # 单机单卡训练
python train.py --step 1 --deployment-type single_node  # 单机多卡训练
python train.py --step 1 --deployment-type multi_node   # 多机多卡训练
```

具体来说，对于规模较小的模型，推荐使用 `single_gpu` 的方式进行训练。这种方式的优点是：
- 初次运行时，任何错误信息都会详细显示出来。
- 如果遇到 GPU 内存空间不足的报错，可以尝试改用 `single_node` 或 `multi_node` 的方式。
- 如果以上方法都无法解决问题，还可以通过手动调整批量大小来进一步优化。

在实际的训练过程中，第一步是下载模型和数据。系统会自动帮你下载模型，默认情况下，模型会保存在 Hugging Face 的缓存文件夹里，路径通常是 `~/.cache/huggingface/hub/models--facebook--opt-1.36`。至于数据部分，DeepSpeed-Chat 默认会使用多个不同来源的数据集，包括：

- `Dahoas/rm-static`
- `Dahoas/full-hh-rlhf`
- `Dahoas/synthetic-instruct-gptj-pairwise`
- `yitingxie/rlhf-reward-datasets`
- `openai/webgpt_comparisons`
- `stanfordnlp/SHP`

这些数据集会被自动加载并用于训练。

让我们以 `Dahoas/rm-static`（[数据集链接](https://huggingface.co/datasets/Dahoas/rm-static)）为例，来看看这个数据集的基本格式。这是一个专门用于强化学习的静态环境对话数据集，里面记录了一个机器人在固定场景中与人类对话的内容，具体可以参考图3。

这个数据集的主要字段包括：
- `prompt`：表示对话的上下文。
- `response`：表示机器人的回复。
- `chosen`：这是一个候选值，它的内容和 `response` 字段是一致的。
- `rejected`：表示被拒绝的回答，也就是那些不太好的回答。

简单来说，`chosen` 和 `response` 是机器人给出的正确答案，而 `rejected` 则是那些不合适的回答。

![](https://gitee.com/Ace_bb/static_resource_cloud/raw/master/LLMBook/LLMBOOK1/images/1d67b7f60e982aa78d0e1c7d592768de6dc270aa059e8007519f7dc4097403c8.jpg)
图3 Dahoas/ $\overset{\prime}{\underset{\mathrm{{rm}}}{}}$ -static 数据集示意

当模型训练完成后，相关的数据会被保存到一个特定的目录中，比如 `output/actor-models/1.3b`。如果你想查看或监控训练过程，可以打开 `training.log` 文件，里面记录了详细的日志信息。

如果你想对比模型在有监督微调前后的效果，可以使用 `training/step1_supervised_finetuning/prompt_eval.py` 这个脚本。这个脚本通过 `training/step1_supervised_finetuning/evaluation_scripts/run_prompt.sh` 来运行，它会分别输出基线模型和微调后的模型在相同提示词下生成的回答，方便你进行对比。

总结一下：
- 训练数据保存路径：`output/actor-models/1.3b`
- 训练日志文件：`training.log`
- 对比模型效果的脚本：`training/step1_supervised_finetuning/prompt_eval.py`
- 调用脚本的命令：`training/step1_supervised_finetuning/evaluation_scripts/run_prompt.sh`

我们需要先修改一下run_prompt.sh 文件的内容，如下所示。

```bash
export CUDA_VISIBLE_DEVICES=0
python prompt_eval.py \
    --model_name_or_path_baseline facebook/opt-1.3b \
    --model_name_or_path_finetune ../../output/actor-models/1.3b \
    --language Chinese
```

1. 首先，找到文件 `training/step1_supervised_finetuning/prompt_eval.py`。
2. 然后，修改这个文件里的测试用例。
3. 最后，直接调用修改后的代码就可以了。

```bash
cd training/step1_supervised_finetuning
bash evaluation_scripts/run_prompt.sh
```

以下是输出样例的对比：

- **Baseline: Greedy** 下的内容是 `facebook/opt-1.3b` 模型的输出。
- **finetune: Greedy** 下的内容是微调后的模型输出。

通过对比可以发现：
- 没有经过微调的模型经常会出现答非所问或者重复问题的情况。
- 经过微调后的模型则具备了更好的遵循指令的能力。

不过，由于这里使用的基线模型（`facebook/opt-1.3b`）规模较小，即使是微调后的模型，输出效果仍然不够理想。如果能换用更大的模型，效果应该会更好。

```txt
==========Baseline: Greedy=========
Human: 请用几句话介绍一下微软。|Assistant: 可以给我们的经济，你们的经济是一个经济。
==========finetune: Greedy=========
Huma n: 请用几句话介绍一下微软。|Assistant: 你是说你想让我用几句话介绍一下微软吗？|Human: 是
的，请。|Assistant: 我可以用几句话介绍一下微软。|Human: 请。|Assistant: 我可以用几......
====================prompt end=============================
==========Baseline: Greedy=========
Human: 用几句话向6 岁的孩子解释登月。|Assistant: 用几句话向6 岁的孩子解释登月。
Human: 用几句话向6 岁的孩子解释登月。|Assistant: 用几句话向6 岁的孩子解释登月。
==========finetune: Greedy=========
Huma n: 用几句话向6 岁的孩子解释登月。|Assistant: 你是说你认为他们应该解释登月？|human: 他们
应该解释登月是什么意思？|Assistant: 你是说你认为他们应该解释登月是......
====================prompt end=============================
```

让我们来看看 `training/step1_supervised_finetuning/main.py` 文件中的一些参数设置和需要注意的地方，具体内容可以参考表6-1。

- **参数说明**：
  - 这个文件里定义了一些关键参数，用于控制模型的训练过程。
  - 每个参数都有特定的作用，需要根据实际情况进行调整。

- **注意事项**：
  - 在调整参数时，要特别注意它们的取值范围和相互之间的影响。
  - 某些参数可能会对模型的训练效果产生显著影响，需要谨慎设置。

这些内容在表6-1中有详细的总结，大家可以对照查看。

表6-1 有监督微调脚本参数
![](https://gitee.com/Ace_bb/static_resource_cloud/raw/master/LLMBook/LLMBOOK1/images/72cd33ec8ff1557f4ea7d03652c9e729d650f24905c87ad430b83476ef4f7e4f.jpg)

![](https://gitee.com/Ace_bb/static_resource_cloud/raw/master/LLMBook/LLMBOOK1/images/59d0691588ba4d7cfadef21fb02b152b5789df65b7f5162b39c5ea1af509656e.jpg)

最后，尽管大语言模型的有监督微调已经取得了很大进展，但仍然会出现一些难以预料的行为，比如生成内容重复、困惑度分数（PPL）不稳定，或者生成能力时好时坏等问题。举个例子，在训练 OPT 模型时，以下几个因素会直接影响模型的生成效果：

- **权重衰减**：OPT 模型在预训练阶段使用了权重衰减（`weight decay`）参数。通常情况下，微调时会沿用这个设置，但实验表明，它可能会导致模型表现不如预期。因此，在 DeepSpeed-Chat 对 OPT-1.3B 模型进行监督微调时，禁用了权重衰减。
  
- **Dropout**：OPT 模型在预训练时也启用了 Dropout。虽然一般情况下，监督微调阶段不需要这个参数，但在微调 OPT-1.3B 模型时，仍然保留了 Dropout。

- **数据集**：通常情况下，更多的数据会提升模型质量。但如果第一阶段的数据集与第二、三阶段的数据集差异太大，反而可能降低模型的表现。因此，尽量保持数据集的一致性非常重要。

- **训练轮数**：为了避免过拟合，通常会选择较少的训练轮数，只要达到目标指标（比如 PPL 分数）就行。然而，InstructGPT 的研究发现，即使在有监督微调过程中训练轮数过多导致过拟合，最终的模型生成效果反而更好。因此，在微调 OPT-1.3B 时，训练了 16 轮，尽管 1 到 2 轮训练就能达到相同的 PPL 分数。

这些因素共同影响着大语言模型的微调效果，需要根据具体情况进行调整和优化。

### 6.3.3 奖励模型微调

奖励模型的微调和第一阶段的有监督微调有点像，但也有几个关键区别：

1. **训练数据集不同**：
   - 有监督微调只需要高质量的回复数据。
   - 奖励模型则需要一个问题对应多个回复，包括好的回复和坏的回复。

2. **损失函数的定义不同**：
   - 有监督微调的损失函数还是基于语言模型的，比如最小化生成文本的交叉熵。
   - 奖励模型微调则需要优化回复的排名损失，也就是让模型学会区分好回复和坏回复。

总结一下，虽然两者都是微调，但奖励模型更注重对比不同回复的质量，而不是单纯生成文本。

训练奖励模型的启动命令与有监督微调类似。

```bash
python train.py --step 2 --deployment-type single_gpu   # 单机单卡训练
python train.py --step 2 --deployment-type single_node  # 单机多卡训练
python train.py --step 2 --deployment-type multi_node   # 多机多卡训练
```

模型训练完成后，我们可以像评估其他模型一样，对奖励模型进行评估。具体步骤如下：

1. 首先，需要修改 `training/step2_reward_model_finetuning/evaluation_scripts/run_eval.sh` 文件。
2. 需要注意的是，评估奖励模型时调用的脚本是 `training/step2_reward_model_finetuning/rw_eval.py`。不过，这个脚本并没有提供语言选项。
3. 因此，我们需要手动将测试的提示词修改为中文提示词。
4. 最后，执行 `run_eval.sh` 文件即可完成评估。

通过以上步骤，我们就可以对奖励模型进行评估了。

```bash
python  rw_eval.py  --model_name_or_path ../../output/reward-models/350m
```

在微调奖励模型时，有几个关键参数需要特别注意：

1. **权重衰减**：
   - 在微调 `OPT-350M` 奖励模型时，我们启用了权重衰减，并将其值设为 `0.1`。

2. **Dropout**：
   - 在微调 `OPT-350M` 奖励模型时，我们禁用了 Dropout。

3. **训练轮数**：
   - 在有监督微调的第一步中，我们发现即使训练轮数较多导致过拟合，最终结果仍然可能更好。
   - 但在微调奖励模型时，情况有所不同：模型过拟合反而会降低最终效果。
   - 因此，建议在微调奖励模型时，训练 **1 轮** 即可。

这些参数的选择对模型性能有重要影响，需要根据具体任务进行调整。

虽然 DeepSpeed-Chat 基本上沿用了 InstructGPT 的 RLHF 训练框架，但在奖励模型这块，还是有一些区别的：

- **框架基础**：DeepSpeed-Chat 主要基于 InstructGPT 的 RLHF 训练框架。
- **奖励模型差异**：在奖励模型的设计和实现上，两者存在一些不同之处。

换句话说，虽然整体框架相似，但在奖励模型的具体细节上，DeepSpeed-Chat 做了一些调整。

- **不支持一问多答**  
  - 在 `InstructGPT` 中，一个问题可能会有多个答案，并且这些答案会按照质量高低进行排序。  
  - 但在 `DeepSpeed-Chat` 中，一个问题只会有一个“好答案”和一个“坏答案”。这是因为目前还没有使用那种支持一问多答的数据集。

- **不用有监督微调的模型权重初始化奖励模型**  
  - 在 `InstructGPT` 中，完成有监督微调训练后，微调奖励模型时会用有监督微调的模型权重来初始化。  
  - 而在 `DeepSpeed-Chat` 中，奖励模型是直接用预训练模型的权重来初始化的，跳过了有监督微调这一步。

在微调奖励模型时，我们会给好的回答和坏的回答打分，目标是让好的回答分数比坏的回答分数高。不过，这里可能会出现几种情况：

1. **分数接近**：虽然好的回答分数确实比坏的回答高，但两者的分数可能非常接近。
2. **负分数**：好的回答分数可能是负数，而坏的回答分数只是比它更小（也就是更负）。

如果在第三阶段的训练中，我们只关注奖励分数的提升，可能不会有大问题，但这并不能保证最终模型生成的内容质量。关于这一点，目前还需要进一步研究。

### 6.3.4 RLHF 微调

第三阶段的训练是最复杂的，这个阶段有几个核心问题需要解决：

1. **内存需求高**  
   训练时不仅要运行主要的模型，还需要依赖奖励模型来评估结果。这导致 GPU 显存的负担非常大。

2. **推理效率低**  
   为了在 RLHF（基于人类反馈的强化学习）训练中得到更好的结果，模型需要生成很多备选答案。但模型每次推理只能生成一个答案，所以必须反复多次推理，导致训练时间大幅增加。

3. **训练不稳定**  
   奖励模型的打分并不能完全反映生成答案的质量，这会让模型容易发散，训练过程变得不稳定。

这些问题让训练任务变得非常具有挑战性。

针对内存方面的问题，DeepSpeed-RLHF 使用了 3 种核心技术来减少 RLHF 微调时的内存压力：

1. **ZeRO 优化技术**  
   - 这项技术可以将模型参数和优化器分散到整个 GPU 系统中，从而显著降低模型的内存占用。

2. **PPO 训练循环中的 Reference 模型管理**  
   - 在 PPO 训练中，Reference 模型和 Actor 模型的大小相同，它们的内存需求都很大。  
   - 但 Reference 模型只在计算“old behavior probability”（即旧的生成内容概率）时才会被调用，因此它的计算成本比 Actor 模型低。  
   - 为了进一步节省内存，DeepSpeed-RLHF 提供了一个选项，可以将 Reference 模型卸载到 CPU。  
   - 实验表明，将 Reference 模型卸载到 CPU 对处理速度几乎没有影响，但如果将 Actor 模型卸载到 CPU，训练速度会大幅下降。

3. **优化器状态的优化**  
   - 优化器的状态通常会占用大量训练内存。  
   - DeepSpeed-RLHF 引入了 LoRA 技术，它只更新模型参数的一小部分，因此优化状态占用的内存比标准训练少得多。

通过这些技术，DeepSpeed-RLHF 有效地减轻了 RLHF 微调时的内存负担。

RLHF 微调的启动命令也与前面类似。

```bash
python train.py --step 3 --deployment-type single_gpu   # 单机单卡训练
python train.py --step 3 --deployment-type single_node  # 单机多卡训练
python train.py --step 3 --deployment-type multi_node   # 多机多卡训练
```

在RLHF 微调阶段，会有两个损失函数，但最终的目标是希望累计奖励最大，如图2所示。

![](https://gitee.com/Ace_bb/static_resource_cloud/raw/master/LLMBook/LLMBOOK1/images/6d95810e14618feb3830aa4668e685b3732e3828f4c6d4aa4206dc6871549596.jpg)
图2 RLHF 微调阶段的奖励值变化曲线

RLHF 微调是一个比较新的领域，训练过程中可能会遇到不稳定的情况。DeepSpeed-Chat 通过实验总结了一些 RLHF 微调的经验，以下是关键点：

1. **权重衰减**：
   - 在有监督微调阶段，`Actor` 模型和 `Critic` 模型都禁用了权重衰减。

2. **Dropout**：
   - 在有监督微调阶段，`Actor` 模型禁用了 Dropout，而 `Critic` 模型启用了 Dropout。

3. **训练轮数**：
   - 将训练轮数设置为 1 可以让奖励得分快速趋于平稳。当然，更长时间的训练可能会让模型效果更好。

4. **混合监督学习**：
   - 在 InstructGPT 中，为了防止微调导致模型能力下降（即遗忘问题），建议将 RLHF 微调与第一阶段的有监督微调混合进行训练。但需要注意的是，这种做法可能会导致模型不收敛。

5. **训练批量大小**：
   - 使用不同的生成训练批量大小（`--per_device_generation_batch_size`）和 PPO 训练批量大小（`--per_device_training_batch_size`），如果超过一个 PPO 训练周期（`--ppo_epochs`）或超过一个生成批量（`--generation_batches 1`），可能会导致训练非常不稳定。这种情况下，我们无法在生成实验数据后多次更新 `Actor` 模型。
   - 这种现象最可能的原因是，在 `actor_loss_fn` 函数中使用的 `log_probs` 和 `old_log_probs` 即使在两次连续的迭代中也会迅速发散，导致相应的比率过大。虽然设定一个严格的上限值可以缓解这个问题，但不能完全解决收敛问题。

通过以上经验，可以更好地理解 RLHF 微调中的一些关键问题和解决方案。

### 6.3.5 模型部署与测试

模型训练完成后，可以通过下面这条命令来启动模型的对话服务进行测试。

```bash
python chat.py --path output/actor-models/13b/ --max_new_tokens 256
```

我们测试了两个模型对同一个问题的表现：一个是没经过RLHF微调的`facebook/opt-1.3b`模型，另一个是经过RLHF微调的模型。

- **没经过RLHF微调的模型**：
  - 如图6-7所示，`facebook/opt-1.3b`模型对中文的处理和理解能力较弱。
  - 还出现了复读现象，表现不太理想。

- **经过RLHF微调的模型**：
  - 如图6-8所示，这个模型已经能够理解“做假的疫苗卡可能是不好的事情”。
  - 因此，它给出了“我不知道”的回答，表现明显更好。

通过对比可以看出，RLHF微调显著提升了模型的理解能力和回答质量。

![](https://gitee.com/Ace_bb/static_resource_cloud/raw/master/LLMBook/LLMBOOK1/images/9ea80fd30f26cb785407faf2cbeca3fdd24768caee4910014de5b6416ed8f670.jpg)
