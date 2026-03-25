"""
损失函数测试

测试各算法的损失函数实现是否正确
"""

import pytest
import torch
import torch.nn.functional as F

from src.algorithms.ppo import compute_gae, compute_ppo_loss
from src.algorithms.dpo import compute_dpo_loss
from src.algorithms.orpo import compute_orpo_loss
from src.algorithms.simpo import compute_simpo_loss


class TestGAE:
    """测试 GAE 优势计算"""
    
    def test_gae_basic(self):
        """测试 GAE 基本功能"""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(0)
        values = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5]).unsqueeze(0)
        dones = torch.tensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(0)
        
        advantages = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        
        assert advantages.shape == rewards.shape
        assert not torch.isnan(advantages).any()
        assert not torch.isinf(advantages).any()
    
    def test_gae_with_dones(self):
        """测试 GAE 在 episode 终止时的行为"""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(0)
        values = torch.tensor([0.5, 1.0, 1.5, 2.0, 0.0]).unsqueeze(0)
        dones = torch.tensor([0.0, 0.0, 1.0, 0.0]).unsqueeze(0)  # 第 3 步终止
        
        advantages = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        
        assert advantages.shape == rewards.shape
        # 终止后的优势应该只依赖于后续奖励
    
    def test_gae_batched(self):
        """测试批量 GAE 计算"""
        batch_size = 4
        seq_len = 10
        
        rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len + 1)
        dones = torch.zeros(batch_size, seq_len)
        
        advantages = compute_gae(rewards, values, dones)
        
        assert advantages.shape == (batch_size, seq_len)


class TestPPOLoss:
    """测试 PPO 损失计算"""
    
    def test_ppo_loss_basic(self):
        """测试 PPO 损失基本功能"""
        old_log_probs = torch.randn(4, 10)
        new_log_probs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)
        
        loss, ratio, clip_fraction = compute_ppo_loss(
            old_log_probs, new_log_probs, advantages, clip_epsilon=0.2
        )
        
        assert loss.dim() == 0  # 标量
        assert ratio.shape == old_log_probs.shape
        assert 0 <= clip_fraction <= 1
    
    def test_ppo_loss_clipping(self):
        """测试 PPO clip 机制"""
        # 创建极端差异的 log 概率
        old_log_probs = torch.zeros(4, 10)
        new_log_probs = torch.ones(4, 10) * 10  # 很大的差异
        advantages = torch.ones(4, 10)
        
        loss, ratio, clip_fraction = compute_ppo_loss(
            old_log_probs, new_log_probs, advantages, clip_epsilon=0.2
        )
        
        # 大部分 ratio 应该被 clip
        assert clip_fraction > 0.5
    
    def test_ppo_loss_no_clip(self):
        """测试没有 clip 的情况"""
        old_log_probs = torch.randn(4, 10)
        new_log_probs = old_log_probs + 0.01  # 很小的差异
        advantages = torch.randn(4, 10)
        
        loss, ratio, clip_fraction = compute_ppo_loss(
            old_log_probs, new_log_probs, advantages, clip_epsilon=0.2
        )
        
        # 应该很少或没有 clip
        assert clip_fraction < 0.1


class TestDPOLoss:
    """测试 DPO 损失计算"""
    
    def test_dpo_loss_basic(self):
        """测试 DPO 损失基本功能"""
        policy_chosen = torch.randn(4)
        policy_rejected = torch.randn(4)
        ref_chosen = torch.randn(4)
        ref_rejected = torch.randn(4)
        
        loss, metrics = compute_dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
        )
        
        assert loss.dim() == 0
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_dpo_loss_perfect(self):
        """测试 DPO 在完美分类时的损失"""
        # chosen 的 log 概率远大于 rejected
        policy_chosen = torch.ones(4) * 10
        policy_rejected = torch.ones(4) * -10
        ref_chosen = torch.zeros(4)
        ref_rejected = torch.zeros(4)
        
        loss, metrics = compute_dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
        )
        
        # 损失应该接近 0
        assert loss.item() < 0.1
        assert metrics['accuracy'] == 1.0
    
    def test_dpo_loss_beta_effect(self):
        """测试 beta 参数的影响"""
        policy_chosen = torch.ones(4)
        policy_rejected = torch.zeros(4)
        ref_chosen = torch.zeros(4)
        ref_rejected = torch.ones(4)
        
        loss_small_beta, _ = compute_dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.01
        )
        loss_large_beta, _ = compute_dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0
        )
        
        # beta 越大，损失差异应该越大
        assert loss_small_beta != loss_large_beta


class TestORPOLoss:
    """测试 ORPO 损失计算"""
    
    def test_orpo_loss_basic(self):
        """测试 ORPO 损失基本功能"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
        rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
        chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss, metrics = compute_orpo_loss(
            chosen_logits, rejected_logits,
            chosen_labels, rejected_labels,
            beta=0.1, lambda_or=0.5
        )
        
        assert loss.dim() == 0
        assert 'accuracy' in metrics
    
    def test_orpo_loss_without_sft(self):
        """测试不包含 SFT 损失的 ORPO"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
        rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
        chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss, metrics = compute_orpo_loss(
            chosen_logits, rejected_logits,
            chosen_labels, rejected_labels,
            beta=0.1, lambda_or=1.0,
            include_sft_loss=False
        )
        
        assert loss.dim() == 0
        assert metrics.get('sft_loss', 0) == 0


class TestSimPOLoss:
    """测试 SimPO 损失计算"""
    
    def test_simpo_loss_basic(self):
        """测试 SimPO 损失基本功能"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        # 模拟模型输出
        chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
        rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
        chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # 获取 log 概率
        chosen_log_probs = torch.gather(
            torch.log_softmax(chosen_logits, dim=-1),
            dim=2,
            index=chosen_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        rejected_log_probs = torch.gather(
            torch.log_softmax(rejected_logits, dim=-1),
            dim=2,
            index=rejected_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        loss, metrics = compute_simpo_loss(
            chosen_log_probs, rejected_log_probs,
            chosen_labels, rejected_labels,
            beta=0.1, gamma=0.5
        )
        
        assert loss.dim() == 0
        assert 'accuracy' in metrics
        assert metrics['gamma'] == 0.5
    
    def test_simpo_loss_gamma_effect(self):
        """测试 gamma 参数的影响"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
        rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
        chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        chosen_log_probs = torch.gather(
            torch.log_softmax(chosen_logits, dim=-1),
            dim=2,
            index=chosen_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        rejected_log_probs = torch.gather(
            torch.log_softmax(rejected_logits, dim=-1),
            dim=2,
            index=rejected_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        loss_small_gamma, _ = compute_simpo_loss(
            chosen_log_probs, rejected_log_probs,
            chosen_labels, rejected_labels,
            beta=0.1, gamma=0.1
        )
        
        loss_large_gamma, _ = compute_simpo_loss(
            chosen_log_probs, rejected_log_probs,
            chosen_labels, rejected_labels,
            beta=0.1, gamma=1.0
        )
        
        # gamma 越大，损失应该越大（更难满足 margin）
        assert loss_large_gamma > loss_small_gamma
    
    def test_simpo_loss_length_norm(self):
        """测试长度归一化的效果"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
        rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
        chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        chosen_log_probs = torch.gather(
            torch.log_softmax(chosen_logits, dim=-1),
            dim=2,
            index=chosen_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        rejected_log_probs = torch.gather(
            torch.log_softmax(rejected_logits, dim=-1),
            dim=2,
            index=rejected_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 使用长度归一化
        loss_with_norm, metrics_with_norm = compute_simpo_loss(
            chosen_log_probs, rejected_log_probs,
            chosen_labels, rejected_labels,
            beta=0.1, gamma=0.5,
            use_length_norm=True
        )
        
        # 不使用长度归一化
        loss_without_norm, metrics_without_norm = compute_simpo_loss(
            chosen_log_probs, rejected_log_probs,
            chosen_labels, rejected_labels,
            beta=0.1, gamma=0.5,
            use_length_norm=False
        )
        
        # 两种方式的损失应该不同
        assert loss_with_norm != loss_without_norm


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
