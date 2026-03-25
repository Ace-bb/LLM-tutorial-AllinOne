"""训练脚本模块"""

from .train_rm import train_reward_model
from .train_ppo import train_ppo
from .train_dpo import train_dpo

__all__ = ["train_reward_model", "train_ppo", "train_dpo"]
