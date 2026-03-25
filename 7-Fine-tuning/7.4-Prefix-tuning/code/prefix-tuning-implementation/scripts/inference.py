#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prefix-tuning 推理脚本

使用示例：
    # 使用 GPT-2 进行文本生成
    python scripts/inference.py \
        --model_path ./output/gpt2-prefix \
        --prompt "人工智能是" \
        --max_length 100

    # 使用 BART 进行文本摘要
    python scripts/inference.py \
        --model_path ./output/bart-prefix \
        --prompt "这是一段需要摘要的长文本..." \
        --task summarization \
        --max_length 50
"""

import argparse
import torch
from transformers import AutoTokenizer
import os
import sys

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prefix_model import PrefixTuningModel, PrefixTuningConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Prefix-tuning 推理脚本")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型路径")
    parser.add_argument("--config_path", type=str, default=None,
                        help="配置文件路径（可选）")
    
    # 输入参数
    parser.add_argument("--prompt", type=str, default="人工智能是",
                        help="输入提示文本")
    parser.add_argument("--max_length", type=int, default=100,
                        help="生成文本的最大长度")
    parser.add_argument("--min_length", type=int, default=10,
                        help="生成文本的最小长度")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k 采样参数")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p 采样参数")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="重复惩罚系数")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="束搜索的束数")
    
    # 任务类型
    parser.add_argument("--task", type=str, default="generation",
                        choices=["generation", "summarization", "translation"],
                        help="任务类型")
    
    return parser.parse_args()


def generate_text(model, tokenizer, prompt, args, device):
    """
    生成文本
    
    Args:
        model: Prefix-tuning 模型
        tokenizer: Tokenizer
        prompt: 输入提示
        args: 参数
        device: 设备
    
    Returns:
        生成的文本
    """
    # Tokenize 输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    
    print(f"输入：{prompt}")
    print(f"输入 token 数：{input_ids.shape[1]}")
    
    # 生成配置
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_beams": args.num_beams,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # 根据任务类型调整参数
    if args.task == "summarization":
        generate_kwargs["max_length"] = min(args.max_length, 150)
        generate_kwargs["min_length"] = max(args.min_length, 30)
        generate_kwargs["num_beams"] = max(args.num_beams, 4)
    elif args.task == "translation":
        generate_kwargs["num_beams"] = max(args.num_beams, 5)
    
    # 生成
    print("正在生成...")
    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def main():
    """主函数"""
    args = parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"错误：模型路径不存在：{args.model_path}")
        return
    
    # 加载配置
    config_path = os.path.join(args.model_path, "prefix_config.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        config = PrefixTuningConfig(**config_dict)
    else:
        # 使用默认配置
        print("未找到配置文件，使用默认配置")
        config = PrefixTuningConfig()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("加载模型...")
    model = PrefixTuningModel.from_pretrained(args.model_path, config)
    model.to(device)
    model.eval()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数：{total_params:,}")
    print(f"可训练参数：{trainable_params:,} (训练时)")
    
    # 生成文本
    print("\n" + "="*50)
    generated_text = generate_text(model, tokenizer, args.prompt, args, device)
    print("="*50)
    print(f"生成结果：{generated_text}")
    print("="*50)


if __name__ == "__main__":
    main()
