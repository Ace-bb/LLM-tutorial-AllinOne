"""
LoRA 单元测试

测试 LoRA 核心实现的功能正确性。

运行测试：
    pytest tests/test_lora.py -v
"""

import torch
import pytest
import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lora_core import (
    LoRALayer,
    LoRALinear,
    LoRAEmbedding,
    apply_lora_to_model,
    count_parameters,
)


class TestLoRALayer:
    """测试 LoRALayer 类"""
    
    def test_initialization(self):
        """测试初始化"""
        layer = LoRALayer(
            in_features=512,
            out_features=1024,
            r=8,
            alpha=16,
            dropout=0.05
        )
        
        assert layer.r == 8
        assert layer.alpha == 16
        assert layer.scaling == 2.0  # alpha / r
        assert layer.lora_A.shape == (8, 512)
        assert layer.lora_B.shape == (1024, 8)
    
    def test_zero_initialization(self):
        """测试 B 矩阵零初始化"""
        layer = LoRALayer(512, 1024, r=8, alpha=16)
        
        # B 矩阵应该初始化为 0
        assert torch.all(layer.lora_B == 0)
        
        # A 矩阵应该是高斯分布
        assert torch.std(layer.lora_A) > 0
    
    def test_forward_pass(self):
        """测试前向传播"""
        layer = LoRALayer(512, 1024, r=8, alpha=16)
        
        # 初始时，由于 B=0，输出应该全为 0
        x = torch.randn(4, 512)
        output = layer(x)
        
        assert output.shape == (4, 1024)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)
    
    def test_forward_after_training(self):
        """测试训练后的前向传播"""
        layer = LoRALayer(512, 1024, r=8, alpha=16, dropout=0.0)
        
        # 模拟训练：更新 B 矩阵
        with torch.no_grad():
            layer.lora_B.fill_(0.1)
        
        x = torch.randn(4, 512)
        output = layer(x)
        
        # 输出不应该全为 0
        assert output.shape == (4, 1024)
        assert not torch.allclose(output, torch.zeros_like(output), atol=1e-6)
    
    def test_merge_weights(self):
        """测试权重合并"""
        layer = LoRALayer(512, 1024, r=8, alpha=16)
        
        # 设置非零权重
        with torch.no_grad():
            layer.lora_A.fill_(1.0)
            layer.lora_B.fill_(1.0)
        
        delta = layer.merge_weights()
        
        # delta 形状应该与原始权重相同
        assert delta.shape == (1024, 512)
        
        # delta = B @ A * scaling = 1 @ 1 * 2 = 2
        assert torch.allclose(delta, torch.ones_like(delta) * 2.0)
    
    def test_param_count(self):
        """测试参数数量计算"""
        layer = LoRALayer(512, 1024, r=8, alpha=16)
        
        # A: 8*512 = 4096
        # B: 1024*8 = 8192
        # 总计：12288
        assert layer.get_param_count() == 12288
    
    def test_dropout(self):
        """测试 Dropout"""
        layer_train = LoRALayer(512, 1024, r=8, alpha=16, dropout=0.5)
        layer_eval = LoRALayer(512, 1024, r=8, alpha=16, dropout=0.5)
        
        x = torch.randn(4, 512)
        
        # 训练模式：有 dropout
        layer_train.train()
        out_train = layer_train(x)
        
        # 评估模式：无 dropout
        layer_eval.eval()
        out_eval = layer_eval(x)
        
        # 由于 B=0，输出都应该是 0
        assert torch.allclose(out_train, torch.zeros_like(out_train), atol=1e-5)
        assert torch.allclose(out_eval, torch.zeros_like(out_eval), atol=1e-5)


class TestLoRALinear:
    """测试 LoRALinear 类"""
    
    def test_initialization(self):
        """测试初始化"""
        layer = LoRALinear(512, 1024, r=8, alpha=16)
        
        assert hasattr(layer, 'base_layer')
        assert hasattr(layer, 'lora_layer')
        assert isinstance(layer.base_layer, torch.nn.Linear)
        assert isinstance(layer.lora_layer, LoRALayer)
    
    def test_freeze_base(self):
        """测试基础权重冻结"""
        layer = LoRALinear(512, 1024, r=8, alpha=16, freeze_base=True)
        
        # 基础层参数不应该需要梯度
        for param in layer.base_layer.parameters():
            assert not param.requires_grad
        
        # LoRA 层参数应该需要梯度
        assert layer.lora_layer.lora_A.requires_grad
        assert layer.lora_layer.lora_B.requires_grad
    
    def test_forward(self):
        """测试前向传播"""
        layer = LoRALinear(512, 1024, r=8, alpha=16)
        
        x = torch.randn(4, 512)
        output = layer(x)
        
        assert output.shape == (4, 1024)
        
        # 由于 LoRA 初始为 0，输出应该等于基础层输出
        with torch.no_grad():
            base_output = layer.base_layer(x)
        assert torch.allclose(output, base_output)
    
    def test_param_efficiency(self):
        """测试参数效率"""
        layer = LoRALinear(512, 1024, r=8, alpha=16)
        
        total, trainable = count_parameters(layer)
        
        # 总参数 = 基础层 + LoRA 层
        base_params = 512 * 1024 + 1024  # weight + bias
        lora_params = 8 * 512 + 1024 * 8  # A + B
        
        assert total == base_params + lora_params
        assert trainable == lora_params
        
        # 可训练参数比例应该很小
        ratio = trainable / total
        assert ratio < 0.05  # < 5%


class TestLoRAEmbedding:
    """测试 LoRAEmbedding 类"""
    
    def test_initialization(self):
        """测试初始化"""
        emb = LoRAEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
            r=8,
            alpha=16
        )
        
        assert emb.lora_A.shape == (8, 512)
        assert emb.lora_B.shape == (1000, 8)
        assert emb.scaling == 2.0
    
    def test_forward(self):
        """测试前向传播"""
        emb = LoRAEmbedding(1000, 512, r=8, alpha=16)
        
        input_ids = torch.randint(0, 1000, (4, 32))  # batch=4, seq_len=32
        output = emb(input_ids)
        
        assert output.shape == (4, 32, 512)


class TestApplyLoRA:
    """测试 apply_lora_to_model 函数"""
    
    def test_apply_to_linear(self):
        """测试应用到线性层"""
        
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(512, 1024)
                self.linear2 = torch.nn.Linear(1024, 10)
                self.relu = torch.nn.ReLU()
        
        model = SimpleModel()
        
        # 应用 LoRA
        model_lora = apply_lora_to_model(
            model,
            r=8,
            alpha=16,
            target_modules=["linear"]
        )
        
        # linear1 和 linear2 应该被替换为 LoRALinear
        assert isinstance(model_lora.linear1, LoRALinear)
        assert isinstance(model_lora.linear2, LoRALinear)
        
        # relu 不应该被替换
        assert isinstance(model_lora.relu, torch.nn.ReLU)
    
    def test_param_count_after_apply(self):
        """测试应用 LoRA 后的参数统计"""
        
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 1024)
        
        model = SimpleModel()
        total_before, trainable_before = count_parameters(model)
        
        model_lora = apply_lora_to_model(model, r=8, alpha=16)
        total_after, trainable_after = count_parameters(model_lora)
        
        # 总参数增加（LoRA 参数）
        assert total_after > total_before
        
        # 可训练参数减少（基础权重冻结）
        assert trainable_after < trainable_before


class TestIntegration:
    """集成测试"""
    
    def test_training_step(self):
        """测试完整的训练步骤"""
        
        # 创建模型
        layer = LoRALinear(512, 1024, r=8, alpha=16, freeze_base=True)
        
        # 创建优化器（只优化 LoRA 参数）
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, layer.parameters()),
            lr=1e-4
        )
        
        # 创建损失函数
        criterion = torch.nn.MSELoss()
        
        # 模拟训练步骤
        x = torch.randn(4, 512)
        target = torch.randn(4, 1024)
        
        # 前向
        output = layer(x)
        loss = criterion(output, target)
        
        # 反向
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 验证损失是有限值
        assert torch.isfinite(loss)
        assert loss.item() > 0
    
    def test_gradient_flow(self):
        """测试梯度流动"""
        
        layer = LoRALayer(512, 1024, r=8, alpha=16)
        
        x = torch.randn(4, 512, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        
        loss.backward()
        
        # 检查梯度
        assert x.grad is not None
        assert layer.lora_A.grad is not None
        assert layer.lora_B.grad is not None
        
        # 梯度应该是有限值
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(layer.lora_A.grad).all()
        assert torch.isfinite(layer.lora_B.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
