"""
LoRA 微调实战脚本
=================

本脚本演示如何使用 HuggingFace PEFT 库对大型语言模型进行 LoRA 微调。

支持：
- 标准 LoRA（全精度）
- QLoRA（4-bit 量化）
- 多种模型架构（Llama、GPT-2、T5 等）
- 指令微调任务

使用方法：
    # 标准 LoRA 微调
    python lora_finetune.py --model_name meta-llama/Llama-2-7b-hf
    
    # QLoRA 微调（节省显存）
    python lora_finetune.py --model_name meta-llama/Llama-2-7b-hf --use_qlora
    
    # 自定义配置
    python lora_finetune.py --model_name meta-llama/Llama-2-7b-hf \
        --lora_r 16 --lora_alpha 32 --learning_rate 2e-4

作者：LoRA 技术文章示例代码
日期：2026-03-16
"""

import os
import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# HuggingFace 库
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from datasets import load_dataset


@dataclass
class ModelArguments:
    """模型相关参数"""
    
    model_name_or_path: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "预训练模型名称或路径"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "模型版本"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "是否信任远程代码"}
    )


@dataclass
class LoRAArguments:
    """LoRA 配置参数"""
    
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA 秩 (rank). 推荐：8-32"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha 参数。推荐：alpha = 2 * r"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout 比例。推荐：0.05-0.1"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={
            "help": "应用 LoRA 的目标模块。"
                    "Llama: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']"
                    "GPT-2: ['c_attn', 'c_proj', 'c_fc']"
        }
    )
    bias: str = field(
        default="none",
        metadata={"help": "bias 训练策略：'none', 'all', 'lora_only'"}
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "是否使用 Rank-Stabilized LoRA"}
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "是否使用 QLoRA (4-bit 量化)"}
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    
    dataset_name: str = field(
        default="yahma/alpaca-cleaned",
        metadata={"help": "训练数据集名称"}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "数据集配置名称"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "数据预处理工作线程数"}
    )


@dataclass
class TrainingArgs(TrainingArguments):
    """训练参数"""
    
    output_dir: str = field(
        default="./lora-output",
        metadata={"help": "输出目录"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "训练轮数。推荐：1-3，避免过拟合"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "每设备训练批次大小"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "梯度累积步数。有效 batch_size = per_device * accumulation"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "学习率。LoRA 推荐：1e-4 ~ 2e-4，8-bit 可用 1e-3"}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "学习率预热比例。推荐：0.03-0.1"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "学习率调度器类型"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "日志记录步数"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "模型保存步数"}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "是否使用混合精度训练"}
    )
    report_to: str = field(
        default="none",
        metadata={"help": "报告工具：'tensorboard', 'wandb', 'none'"}
    )


def create_quantization_config(use_qlora: bool) -> Optional[BitsAndBytesConfig]:
    """
    创建量化配置（用于 QLoRA）
    
    QLoRA 核心配置：
    - 4-bit 量化：将模型权重从 16-bit 压缩到 4-bit
    - NF4 格式：Normal Float 4，专为神经网络优化的量化格式
    - 双重量化：进一步压缩量化常数
    """
    if not use_qlora:
        return None
    
    return BitsAndBytesConfig(
        load_in_4bit=True,                    # 启用 4-bit 量化
        bnb_4bit_quant_type="nf4",            # 使用 NF4 量化格式
        bnb_4bit_compute_dtype=torch.float16, # 计算时使用 float16
        bnb_4bit_use_double_quant=True,       # 启用双重量化（额外压缩）
        llm_int8_threshold=6.0,               # int8 阈值
    )


def create_lora_config(args: LoRAArguments) -> LoraConfig:
    """
    创建 LoRA 配置
    
    关键参数说明：
    - r: 秩，控制低秩矩阵的大小。r 越大，可训练参数越多
    - alpha: 缩放因子。实际缩放比例为 alpha/r
    - target_modules: 要应用 LoRA 的模块。越多模块，性能越好但显存占用越高
    """
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        task_type=TaskType.CAUSAL_LM,        # 因果语言模型任务
        inference_mode=False,                 # 训练模式
        use_rslora=args.use_rslora,          # 是否使用秩稳定 LoRA
        init_lora_weights="gaussian",         # 高斯初始化
    )


def load_and_preprocess_data(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    num_workers: int = 4
):
    """
    加载并预处理数据集
    
    这里以 Alpaca 指令微调数据集为例。
    实际使用时可根据需求替换为自定义数据集。
    """
    print(f"加载数据集：{dataset_name}")
    
    # 加载数据集
    dataset = load_dataset(dataset_name, split="train")
    
    # 定义预处理函数
    def preprocess_function(examples):
        """
        将指令数据转换为模型输入格式
        
        Alpaca 格式：
        {
            "instruction": "...",
            "input": "...",      # 可选
            "output": "..."
        }
        """
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        ):
            # 构建完整的 prompt
            if input_text:
                text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
            else:
                text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
            texts.append(text)
        
        # 分词
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None
        )
        
        # 添加标签（用于语言建模）
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # 应用预处理
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset.column_names
    )
    
    print(f"数据集大小：{len(tokenized_dataset)} 条样本")
    return tokenized_dataset


def print_trainable_parameters(model):
    """
    打印模型的可训练参数统计
    
    这是 LoRA 微调的重要指标，用于验证参数效率。
    """
    trainable_params = 0
    all_param = 0
    
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"\n{'='*60}")
    print(f"模型参数统计")
    print(f"{'='*60}")
    print(f"总参数量：    {all_param:,}")
    print(f"可训练参数：  {trainable_params:,}")
    print(f"参数效率：    {100 * trainable_params / all_param:.4f}%")
    print(f"{'='*60}\n")


def main():
    """主训练函数"""
    
    # 解析参数
    parser = HfArgumentParser((
        ModelArguments,
        LoRAArguments,
        DataArguments,
        TrainingArgs
    ))
    
    if len(os.sys.argv) == 1:
        # 无参数时使用默认值
        model_args = ModelArguments()
        lora_args = LoRAArguments()
        data_args = DataArguments()
        training_args = TrainingArgs()
    else:
        model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("\n" + "=" * 60)
    print("LoRA 微调训练开始")
    print("=" * 60)
    print(f"模型：{model_args.model_name_or_path}")
    print(f"LoRA 秩：r={lora_args.lora_r}, alpha={lora_args.lora_alpha}")
    print(f"量化：{'QLoRA (4-bit)' if lora_args.use_qlora else '标准 LoRA'}")
    print("=" * 60 + "\n")
    
    # 1. 加载 tokenizer
    print("1. 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   设置 pad_token 为：{tokenizer.pad_token}")
    
    # 2. 加载模型
    print("2. 加载模型...")
    quantization_config = create_quantization_config(lora_args.use_qlora)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        quantization_config=quantization_config,
        device_map="auto",              # 自动分配设备
        torch_dtype=torch.float16,      # 使用 float16 节省显存
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,  # FlashAttention 加速
    )
    
    # 3. 准备模型用于 k-bit 训练（QLoRA 必需）
    if lora_args.use_qlora:
        print("3. 准备模型用于 4-bit 训练...")
        model = prepare_model_for_kbit_training(model)
    
    # 4. 创建 LoRA 配置
    print("4. 创建 LoRA 配置...")
    lora_config = create_lora_config(lora_args)
    print(f"   目标模块：{lora_args.target_modules}")
    
    # 5. 应用 LoRA 到模型
    print("5. 应用 LoRA 到模型...")
    model = get_peft_model(model, lora_config)
    
    # 打印参数统计
    print_trainable_parameters(model)
    
    # 6. 加载数据
    print("6. 加载并预处理数据...")
    train_dataset = load_and_preprocess_data(
        dataset_name=data_args.dataset_name,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        num_workers=data_args.preprocessing_num_workers
    )
    
    # 7. 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型不使用 MLM
    )
    
    # 8. 创建 Trainer
    print("7. 创建 Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 9. 开始训练
    print("8. 开始训练...")
    print(f"   输出目录：{training_args.output_dir}")
    print(f"   训练轮数：{training_args.num_train_epochs}")
    print(f"   学习率：{training_args.learning_rate}")
    print(f"   有效批次大小：{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print()
    
    trainer.train()
    
    # 10. 保存模型
    print("9. 保存模型...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"\n训练完成！模型保存到：{training_args.output_dir}")
    print(f"LoRA 权重文件大小：约 {os.path.getsize(os.path.join(training_args.output_dir, 'adapter_model.safetensors')) / 1024 / 1024:.2f} MB")
    
    # 11. 导出合并后的模型（可选）
    print("\n10. 导出合并后的模型（可选）...")
    print("    如需合并 LoRA 权重到基础模型，使用以下代码：")
    print("""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, "./lora-output")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("./merged-model")
    """)


if __name__ == "__main__":
    main()
