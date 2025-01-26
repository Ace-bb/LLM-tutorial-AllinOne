
LLaMA 模型是目前最流行和性能最强大的开源模型之一，基于LLaMA 所构造的模型生态可以覆盖绝大部分模型使用场景。在设置完必要的数据和环境配置后，本节将逐步演示如何使用DeepSpeed 框架训练LLaMA 模型。  

Deepspeed 可以很好地兼容PyTorch 和CUDA 的大多数版本，其安装过程通常无需指定特殊配置选项，可以直接通过pip 命令完成。  

# 1. 训练数据配置  

使用PyTorch 和Transformers 库来设置预训练模型的数据加载器，以实现在单机或多机分布式训练环境中对数据的加载和采样。需要导入的模块为：  

• DataLoader: PyTorch 提供的工具，用于从数据集加载数据到模型进行训练或评估。  
• RandomSampler 和SequentialSampler: 这是PyTorch 提供的两种采样器。RandomSampler 随机采样数据，而SequentialSampler 顺序采样数据。  
• DistributedSampler：用于分布式训练的数据采样器。  
• default_data_collator: Transformers 库提供的默认数据收集器，用于将多个样本整合为一个批量数据。  
• create_pretrain_dataset: 一个自定义函数，用于创建预训练数据集。  

通过检查args.local_rank 是否为−1，代码决定使用普通的采样器（单机）还是分布式采样器（多机）。DistributedSampler 确保在分布式训练环境中，每个进程或节点都能获得数据的一个不重复的子集，这使得分布式训练变得可能。而在单机环境中，使用常规的随机或顺序采样器即可。具体代码如下所示：  

```python
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import default_data_collator
from utils.data.data_utils import create_pretrain_dataset
#数据准备
train_dataset,eval_dataset =create_pretrain_dataset(
    args.local_rank,
    args.data_path,
    args.data_split,
    args.data_output_path,
    args.seed,
    tokenizer,
    args.max_seq_len)
#DataLoaders创建:
if args.local_rank ==-1:
    train_sampler= RandomSampler(train_dataset)
    eval_sampler= SequentialSampler(eval_dataset)
else:
    train_sampler= DistributedSampler(train_dataset)
    eval_sampler= DistributedSampler(eval_dataset)
train_dataloader=DataLoader(train_dataset,
    collate_fn=default_data_collator,
    sampler=train_sampler,
    batch_size=args.per_device_train_batch_size)
eval_dataloader =DataLoader(eval_dataset,
    collate_fn=default_data_collator,
    sampler=eval_sampler,
    batch_size=args.per_device_eval_batch_size)

```

# 2. 模型载入  

使用Transformers 库加载和配置LLaMA 模型及其相关的分词器。在从transformers 库中导入LLaMA 模型、相应的分词器和模型配置后，使用from_pretrained 方法来加载预训练的LLaMA 模型、分词器和配置。为了确保分词器可以处理各种文本长度，还需要进行填充设置。如果分词器还没有指定填充符号，将其设置为[PAD]，并确定填充行为发生在句子的右侧。此外，为了保证模型能够正确地处理句子结束和填充，还为模型配置设置了结束符号和填充符号的 $\mathrm{ID}$ 。最后，为了优化模型在硬件上的性能，还需要调整模型的词汇表嵌入大小，使其成为8 的倍数。通过这些步骤，可以成功地加载并配置LLaMA 模型，为后续的训练任务做好了准备。具体代码如下：  

```python
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
 # 载入分词器：将获得正确的分词器并根据模型系列设置填充词元。
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)
if tokenizer.pad_token is None:
    # 断言 tokenizer.eos_token 不为 None。
    # 向分词器中加入特殊词元。
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'right'
model_config = LlamaConfig.from_pretrained(model_name_or_path)
model = LlamaForCausalLM.from_pretrained(model_name_or_path, config=model_config)
model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(int(
    8 *
    math.ceil(len(tokenizer) / 8.0))) # make the vocab size multiple of 8
```

# 3. 优化器设置  

DeepSpeed 库提供了高效的优化器算法，如DeepSpeedCPUAdam 和FusedAdam，这些算法经过特殊优化以提高在大规模数据和模型上的训练速度。优化器配置主要包含一下几个方面：  

• 参数分组：通过get_optimizer_grouped_parameters 函数将模型参数分为两组：一组使用权重衰减，另一组则不使用。这种参数分组有助于正则化模型，防止过拟合，并允许对特定参数应用不同的学习设置。  
• 优化器选择：根据训练设置（如是否在CPU 上进行模型参数卸载），我们可以选择使用Deep-SpeedCPUAdam 或FusedAdam 优化器。这两种优化器都是对经典的Adam 优化器进行优化和改进的版本，为大规模训练提供了高效性能。  
• 学习率调度：不同于固定的学习率，学习率调度器在训练过程中动态调整学习率。例如，在训练初期快速提高学习率以加速收敛，然后在训练中后期逐渐降低学习率以获得更精细的优化。我们的配置考虑了预热步骤、训练的总步数以及其他关键因素。具体代码如下所示：  
```python
from transformers import get_scheduler
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# 设置需要优化的模型参数以及优化器。
optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    model, args.weight_decay, args.learning_rate)

AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
optimizer = AdamOptimizer(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          betas=(0.9, 0.95))

num_update_steps_per_epoch = math.ceil(
    len(train_dataloader) / args.gradient_accumulation_steps)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
)

def get_optimizer_grouped_parameters(model,
                                    weight_decay,
                                    no_decay_name_list=[
                                        "bias", "LayerNorm.weight"
                                    ]):
    # 将权重分为两组，一组有权重衰减，另一组没有。
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters
```
# 4. DeepSpeed 设置  

在配置代码的开始，定义了两个关键参数：GLOBAL_BATCH_SIZE: 定义了全局的批次大小。这通常是所有GPU 加起来的总批次大小。MICRO_BATCH_SIZE: 定义了每个GPU 上的微批次大小。微批次处理可以帮助大型模型在有限的GPU 内存中运行，因为每次只加载并处理一小部分数据。训练配置函数get_train_ds_config 主要包括以下内容：  

• ZeRO 优化配置：ZeRO（Zero Redundancy Optimizer）是DeepSpeed 提供的一种优化策略，旨在减少训练中的冗余并加速模型的训练。其中的参数，如offload_param 和offload_optimizer，允许用户选择是否将模型参数或优化器状态卸载到CPU。  
• 混合精度训练：通过设置fp16 字段，使得模型可以使用16 位浮点数进行训练，从而加速训练过程并减少内存使用。  
• 梯度裁剪：通过gradient_clipping 字段，我们可以防止训练过程中的梯度爆炸问题。  
• 混合引擎配置：hybrid_engine 部分允许用户配置更高级的优化选项，如输出分词的最大数量和推理张量的大小。  
• TensorBoard 配置：使用DeepSpeed 时，可以通过配置选项直接集成TensorBoard，从而更方便地跟踪训练过程。  

验证集配置函数：get_eval_ds_config：此函数提供了DeepSpeed 的验证集。与训练配置相比，验证集配置更为简洁，只需要关注模型推理阶段即可。  

具体代码如下所示：  
```python
import torch
import deepspeed.comm as dist

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4

def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        tb_path="",
                        tb_name=""):
    # 设置训练过程的 DeepSpeed 配置。
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }

    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }

def get_eval_ds_config(offload, stage=0):
    # 设置评价过程的 DeepSpeed 配置。
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
```

# 5. DeepSpeed 初始化  

在设置DeepSpeed 配置参数后，可以利用DeepSpeed 进行模型训练的初始化，初始化流程包括：• 确定运行的设备：首先，代码检查是否有指定的本地GPU（通过args.local_rank）。如果没有指定，程序默认使用CUDA 设备。否则，它会为进程设置指定的GPU。  

• 初始化分布式后端：在分布式训练中，使用deepspeed.init_distributed() 函数实现每个进程与其他进程的同步，初始化分布式环境。  

• 获取当前进程的全局排序：在分布式训练中，使用torch.distributed.get_rank() 获得每个进程的唯一排序或ID。  

• 设置DeepSpeed 配置：根据用户参数（如是否进行offload、使用哪个zero stage 等），构建了一个DeepSpeed 配置字典，来决定训练设置。  

• 同步所有工作进程：使用torch.distributed.barrier() 确保在进一步的初始化之前所有进程都已同步。  

• DeepSpeed 初始化：这是最关键的一步。通过deepspeed.initialize，可以将模型、优化器、参数和先前构建的DeepSpeed 配置传递给库，进行初始化。这个函数会返回一个已经根据DeepSpeed配置进行了优化的模型和优化器。  

• 梯度检查点：对于特别大的模型，梯度检查点是一种节省显存的技巧，即只在需要时计算模型的中间梯度。如果用户启用了这个选项，则会调用model.gradient_checkpointing_enable() 方法来实现相关功能。  
具体代码如下所示：  

```python
import deepspeed

if args.local_rank == -1:
    device = torch.device("cuda")
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

# 初始化分布式后端，它将负责同步节点 / GPU。
torch.distributed.init_process_group(backend='nccl')
deepspeed.init_distributed()

args.global_rank = torch.distributed.get_rank()

ds_config = get_train_ds_config(offload=args.offload,
                                stage=args.zero_stage,
                                enable_tensorboard=args.enable_tensorboard,
                                tb_path=args.tensorboard_path,
                                tb_name="step1_model")

ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

# 设置训练种子。
set_random_seed(args.seed)

torch.distributed.barrier()

# 使用 DeepSpeed 对模型、优化器进行初始化。
model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    args=args,
    config=ds_config,
    lr_scheduler=lr_scheduler,
    dist_init_required=True)

if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
```

# 6. 模型训练  

借助DeepSpeed 框架实现对模型的训练，训练步骤大致分为以下几个阶段：  

• 训练前的准备：使用print_rank_0 函数输出当前的训练状态。该函数确保只有指定的进程（通常是主进程）会打印消息，避免了多进程环境下的重复输出。在开始训练之前，对模型进行了一次评估，计算模型的困惑度。  
• 训练循环：每个周期的开始，都会打印当前周期和总周期数。在每次迭代中，数据批次首先被移动到相应的GPU 设备，接着模型对这个批次进行前向传播计算损失。使用model.backward(loss)计算梯度，并使用model.step() 更新模型参数。对于主进程，还会使用print_throughput 函数打印吞吐量，这有助于了解模型的训练速度和效率。  
• 保存模型：如果指定了输出目录，模型的状态和配置将被保存。模型可以在不同的格式中保存，例如Hugging Face 的模型格式或DeepSpeed 的Zero Stage 3 特定格式。save_hf_format函数用于保存模型为Hugging Face 格式，这意味着训练后的模型可以使用Hugging Face 的from_pretrained 方法直接加载。对于Zero Stage 3，save_zero_three_model 函数负责保存，因为在这个阶段，每个GPU 只保存了模型的一部分。  
具体代码如下所示：  
```python
# 模型训练部分。
print_rank_0("***** Running training *****", args.global_rank)
print_rank_0(
    f"***** Evaluating perplexity, "
    f"Epoch {0}/{args.num_train_epochs} *****",
    args.global_rank)
perplexity = evaluation(model, eval_dataloader)
print_rank_0(f"ppl: {perplexity}", args.global_rank)

for epoch in range(args.num_train_epochs):
    print_rank_0(
        f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, "
        f"Total Micro Batches {len(train_dataloader)}",
        args.global_rank)
    model.train()
    import time
    for step, batch in enumerate(train_dataloader):
        start = time.time()
        batch = to_device(batch, device)
        outputs = model(*batch, use_cache=False)
        loss = outputs.loss
        if args.print_loss:
            print(
                f"Epoch: {epoch}, Step: {step}, "
                f"Rank: {torch.distributed.get_rank()}, loss = {loss}"
            )
        model.backward(loss)
        model.step()
        end = time.time()
        if torch.distributed.get_rank() == 0:
            print_throughput(model.model, args, end - start,
                             args.global_rank)

if args.output_dir is not None:
    print_rank_0('saving the final model ...', args.global_rank)
    model = convert_lora_to_linear_layer(model)

if args.global_rank == 0:
    save_hf_format(model, tokenizer, args)

if args.zero_stage == 3:
    # 对于 Zero 阶段 3，每个 GPU 只有模型的一部分，因此我们需要一个特殊的保存函数。
    save_zero_three_model(model,
                         args.global_rank,
                         args.output_dir,
                         zero_stage=args.zero_stage)

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

# 此函数仅用于打印 Zero 阶段 1 和 2 的吞吐量。
def print_throughput(hf_model, args, e2e_time, rank=0):
    if rank <= 0:
        hf_config = hf_model.config
        num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)

        gpus_per_model = torch.distributed.get_world_size()
        seq_length = args.max_seq_len
        batch_size = args.per_device_train_batch_size
        samples_per_second = batch_size / e2e_time
        checkpoint_activations_factor = 4 if args.gradient_checkpointing else 3
        if args.lora_dim > 0:
            k = args.lora_dim * 2 / hidden_size
            checkpoint_activations_factor -= (1 - k)

        hf_model._num_params = sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
            for p in hf_model.parameters()
        ])
        params_in_billions = hf_model._num_params / (1e9)

        # Megatron 论文中计算训练 FLOPs 的公式。
        train_flops_per_iteration = calculate_flops(
            checkpoint_activations_factor, batch_size, seq_length, hf_config)

        train_tflops = train_flops_per_iteration / ((e2e_time * gpus_per_model) * (10**12))

        param_string = f"{params_in_billions:.3f} B" if params_in_billions != 0 else "NA"
        print(
            f"Model Parameters: {param_string}, Latency: {e2e_time:.2f}s, "
            f"TFLOPs: {train_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, "
            f"Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, "
            f"Sequence Length: {seq_length}"
        )

def save_hf_format(model, tokenizer, args, sub_folder=""):
    # 用于保存 Hugging Face 格式，以便我们可以在 hf.from_pretrained 中使用它。
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema, 'module') else model_ema

    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]), enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict

```

# 4.5 实践思考  

大语言模型训练过程需要花费大量计算资源，LLaMA-2 70B 模型训练时间为172 万GPU 小时，使用1024 卡A100 集群，需要花费70 天时间。分布式系统性能优化对于大语言模型训练就显得尤为重要。大语言模型训练所使用的高性能计算集群大都采用包含8 卡A100 80GB SXM 或者H100 80GB SXM 的终端，服务器之间采用400Gb 以上的高速InfiniBand 网络，采用胖树网络结构。2023 年5 月，NVIDIA 发布了DGX GH200 超级计算机，使用NVLink Switch 系统，将256个GH200 Grace Hopper 芯片和144TB 的共享内存连接成一个计算单元，为更大规模的语言模型训练提供了硬件基础。  

DeepSpeed[138]、Megatron-LM[135]、Colossal-AI[144] 等多种分布式训练框架都可以用于大语言模型训练。由于目前大多数开源语言模型都是基于Huggingface Transformers 开发，因为在分布式架构选择上需要考虑Huggingface Transformers 的匹配。上述三种分布式架构对于HuggingfaceTransformers 支持都较为方便。此外，千亿以上大规模语言模型训练需要混合数据并行、流水线并行以及张量并行，其中张量并行需要对原始模型代码进行一定程度的修改。针对参数量300 亿以下的模型，可以不使用张量并行，使用目前的分布式训练框架几乎可以不修改代码就可以实现多机多卡分布式训练。  

大语言模型训练主要超参数包括批次大小（Batch Size）、学习率（Learning Rate）、优化器（Optimizer）。这些超参数的设置对于大语言模型稳定训练非常重要，常常会出现训练不稳定的问题，很容易导致模型崩溃。对于批次大小设定，不同的模型所使用数值差距很大，LLaMA-2 中使用的全局批次大小为4M 词元，而在GPT-3 训练中GPT-3 的批大小从32K 逐渐增加到3.2M 个词元。针对学习率调度策略，现有的大语言模型通常都引入热身（Warm-up）和衰减（Decay）策略。在训练的初始阶段（通常是训练量的 $0.1\%$ 到 $0.5\%$ ）采用线性热身调度逐渐增加学习率，将其提高到最大值，最大值的范围大约在 $5\times10^{-5}$ 到 $1\times10^{-4}$ 之间。此后，采用余弦衰减策略，逐渐将学习率降低到其最大值的约 $10\%$ ，直到训练损失收敛。大语言模型训练通常使用 $\mathrm{Adam}^{[145]}$ 或AdamW 优化器[146], 其所使用超参数设置通常为 $\beta_{1}=0.9$ ， $\beta_{2}=0.95$ ， $\epsilon=10^{-8}$ 。此外，为了稳定训练还需要使用权重衰减（Weight Decay）和梯度裁剪（Gradient Clipping）方法，梯度裁剪的阈值通常设置为1.0，权重衰减率设置为0.1。 