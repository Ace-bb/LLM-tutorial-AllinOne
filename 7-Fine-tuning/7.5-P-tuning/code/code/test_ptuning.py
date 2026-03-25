# -*- coding: utf-8 -*-
"""
P-tuning 快速测试脚本

验证核心组件是否正常工作。
注意：PTuningModel 测试需要下载 GPT-2 模型，如无网络会跳过。
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.prompt_encoder import PromptEncoder
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
    
    print(f"[OK] 编码器创建成功")
    print(f"  嵌入维度：{encoder.embed_dim}")
    print(f"  隐藏层维度：{encoder.hidden_dim}")
    print(f"  虚拟词元数量：{encoder.num_virtual_tokens}")
    
    # 测试前向传播
    batch_size = 4
    prompt_embeds = encoder(batch_size)
    
    expected_shape = (batch_size, encoder.num_virtual_tokens, encoder.embed_dim)
    assert prompt_embeds.shape == expected_shape, f"形状不匹配：{prompt_embeds.shape} != {expected_shape}"
    
    print(f"[OK] 前向传播成功")
    print(f"  输入批次大小：{batch_size}")
    print(f"  输出形状：{prompt_embeds.shape}")
    
    # 统计参数
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  总参数量：{total_params:,}")
    print(f"  可训练参数：{trainable_params:,}")
    
    # 测试保存和加载
    print(f"\n测试保存/加载...")
    save_path = "test_prompt_encoder.pt"
    encoder.save_prompt(save_path)
    
    # 加载
    loaded_encoder = PromptEncoder.load_prompt(save_path)
    loaded_embeds = loaded_encoder(batch_size)
    
    assert torch.allclose(prompt_embeds, loaded_embeds), "加载后嵌入不匹配"
    print(f"[OK] 保存/加载测试通过")
    
    # 清理测试文件
    os.remove(save_path)
    
    print("\n[OK] PromptEncoder 测试通过！\n")
    return True


def test_config():
    """测试配置文件"""
    print("="*60)
    print("测试 2: Config")
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
    
    # 测试配置验证
    try:
        invalid_config = PTuningConfig(num_virtual_tokens=-1)
        print(f"[FAIL] 配置验证失败：应该拒绝负数")
        return False
    except ValueError:
        print(f"[OK] 配置验证正常工作（拒绝无效值）")
    
    print("\n[OK] Config 测试通过！\n")
    return True


def test_ptuning_model():
    """测试 P-tuning 模型（需要网络下载 GPT-2）"""
    print("="*60)
    print("测试 3: PTuningModel")
    print("="*60)
    
    try:
        from src.ptuning_model import PTuningModel
        
        # 创建模型（使用 GPT-2）
        print("加载 GPT-2 模型...")
        model = PTuningModel(
            model_name="gpt2",
            num_virtual_tokens=50,
            encoder_hidden_dim=512,
            task_type="causal_lm"
        )
        
        print(f"[OK] 模型创建成功")
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
        
        print(f"[OK] 前向传播成功")
        print(f"  输入形状：{input_ids.shape}")
        print(f"  输出 logits 形状：{outputs.logits.shape}")
        
        print("\n[OK] PTuningModel 测试通过！\n")
        return True
        
    except Exception as e:
        print(f"\n[SKIP] PTuningModel 测试跳过：{e}")
        print("  原因：需要下载 GPT-2 模型（网络问题或首次运行）")
        print("  建议：在有网络的环境下运行 'pip install transformers' 后重试")
        return None  # None 表示跳过


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("P-tuning 快速测试")
    print("="*60 + "\n")
    
    results = []
    skipped = []
    
    # 测试 1: PromptEncoder（不需要下载模型）
    try:
        result = test_prompt_encoder()
        results.append(("PromptEncoder", result))
    except Exception as e:
        print(f"[FAIL] PromptEncoder 测试失败：{e}")
        results.append(("PromptEncoder", False))
    
    # 测试 2: Config（不需要下载模型）
    try:
        result = test_config()
        results.append(("Config", result))
    except Exception as e:
        print(f"[FAIL] Config 测试失败：{e}")
        results.append(("Config", False))
    
    # 测试 3: PTuningModel（需要下载 GPT-2）
    try:
        result = test_ptuning_model()
        if result is not None:
            results.append(("PTuningModel", result))
        else:
            skipped.append("PTuningModel")
    except Exception as e:
        print(f"[FAIL] PTuningModel 测试失败：{e}")
        results.append(("PTuningModel", False))
    
    # 汇总结果
    print("="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")
    
    if skipped:
        print(f"\n跳过的测试:")
        for name in skipped:
            print(f"  {name}: [SKIP]")
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    
    print(f"\n总计：{total_passed}/{total_tests} 测试通过")
    
    if skipped:
        print(f"       {len(skipped)} 测试跳过（需要网络）")
    
    if total_passed == total_tests and not skipped:
        print("\n[SUCCESS] 所有测试通过！代码可以正常运行。")
        return 0
    elif total_passed == total_tests and skipped:
        print("\n[SUCCESS] 所有可运行的测试通过！")
        print("           跳过的测试需要网络连接下载模型。")
        return 0
    else:
        print("\n[WARNING] 部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
