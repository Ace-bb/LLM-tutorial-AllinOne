# -*- coding: utf-8 -*-
"""
P-tuning 快速测试脚本

验证核心组件是否正常工作。
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.prompt_encoder import PromptEncoder
from src.ptuning_model import PTuningModel
from config import PTuningConfig


def test_prompt_encoder():
    """测试提示编码器"""
    print("="*60)
    print("测试 1: PromptEncoder")
    print("="*60)
    
    # 创建编码器（GPT-2 配置）
    encoder = PromptEncoder(
        embed_dim=768,
        hidden_dim=512,
        num_virtual_tokens=50,
        bidirectional=True
    )
    
    print(f"✓ 编码器创建成功")
    print(f"  嵌入维度：{encoder.embed_dim}")
    print(f"  隐藏层维度：{encoder.hidden_dim}")
    print(f"  虚拟词元数量：{encoder.num_virtual_tokens}")
    
    # 测试前向传播
    batch_size = 4
    prompt_embeds = encoder(batch_size)
    
    expected_shape = (batch_size, encoder.num_virtual_tokens, encoder.embed_dim)
    assert prompt_embeds.shape == expected_shape, f"形状不匹配：{prompt_embeds.shape} != {expected_shape}"
    
    print(f"✓ 前向传播成功")
    print(f"  输入批次大小：{batch_size}")
    print(f"  输出形状：{prompt_embeds.shape}")
    
    # 统计参数
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  参数量：{total_params:,}")
    
    print("\n✓ PromptEncoder 测试通过！\n")
    return True


def test_ptuning_model():
    """测试 P-tuning 模型"""
    print("="*60)
    print("测试 2: PTuningModel")
    print("="*60)
    
    try:
        # 创建模型（使用 GPT-2）
        print("加载 GPT-2 模型...")
        model = PTuningModel(
            model_name="gpt2",
            num_virtual_tokens=50,
            encoder_hidden_dim=512,
            task_type="causal_lm"
        )
        
        print(f"✓ 模型创建成功")
        print(f"  基础模型：gpt2")
        print(f"  虚拟词元数量：{model.num_virtual_tokens}")
        
        # 打印参数统计
        print(f"\n参数统计:")
        model.print_trainable_parameters()
        
        # 测试前向传播
        print(f"\n测试前向传播...")
        batch_size = 2
        seq_len = 32
        
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        expected_logits_shape = (batch_size, seq_len, 50257)
        assert outputs.logits.shape == expected_logits_shape, \
            f"形状不匹配：{outputs.logits.shape} != {expected_logits_shape}"
        
        print(f"✓ 前向传播成功")
        print(f"  输入形状：{input_ids.shape}")
        print(f"  输出 logits 形状：{outputs.logits.shape}")
        
        print("\n✓ PTuningModel 测试通过！\n")
        return True
        
    except Exception as e:
        print(f"\n✗ PTuningModel 测试失败：{e}")
        print("  （可能是网络问题导致模型下载失败，这是正常的）")
        return False


def test_config():
    """测试配置文件"""
    print("="*60)
    print("测试 3: Config")
    print("="*60)
    
    from config import default_config, get_task_config
    
    print(f"默认配置:")
    print(f"  虚拟词元数量：{default_config.num_virtual_tokens}")
    print(f"  LSTM 隐藏层：{default_config.encoder_hidden_dim}")
    print(f"  学习率：{default_config.learning_rate}")
    
    # 测试任务配置
    sentiment_config = get_task_config("sentiment_classification")
    print(f"\n情感分类任务配置:")
    print(f"  虚拟词元数量：{sentiment_config.num_virtual_tokens}")
    print(f"  训练轮数：{sentiment_config.num_epochs}")
    
    print("\n✓ Config 测试通过！\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("P-tuning 快速测试")
    print("="*60 + "\n")
    
    results = []
    
    # 测试 1: PromptEncoder（不需要下载模型）
    try:
        results.append(("PromptEncoder", test_prompt_encoder()))
    except Exception as e:
        print(f"✗ PromptEncoder 测试失败：{e}")
        results.append(("PromptEncoder", False))
    
    # 测试 2: Config（不需要下载模型）
    try:
        results.append(("Config", test_config()))
    except Exception as e:
        print(f"✗ Config 测试失败：{e}")
        results.append(("Config", False))
    
    # 测试 3: PTuningModel（需要下载 GPT-2）
    try:
        results.append(("PTuningModel", test_ptuning_model()))
    except Exception as e:
        print(f"✗ PTuningModel 测试失败：{e}")
        results.append(("PTuningModel", False))
    
    # 汇总结果
    print("="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    
    print(f"\n总计：{total_passed}/{total_tests} 测试通过")
    
    if total_passed == total_tests:
        print("\n🎉 所有测试通过！代码可以正常运行。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
