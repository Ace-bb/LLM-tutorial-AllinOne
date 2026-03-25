"""
Prefix-tuning Implementation Package

This package provides a complete implementation of Prefix-tuning,
a parameter-efficient fine-tuning method for pretrained language models.

Author: AI Assistant
Date: 2026-03-16
"""

from .prefix_model import PrefixTuningModel, PrefixTuningConfig
from .trainer import PrefixTrainer
from .utils import (
    load_model_and_tokenizer,
    save_prefix,
    load_prefix,
    set_seed,
    compute_metrics,
)

__version__ = "1.0.0"
__all__ = [
    "PrefixTuningModel",
    "PrefixTuningConfig",
    "PrefixTrainer",
    "load_model_and_tokenizer",
    "save_prefix",
    "load_prefix",
    "set_seed",
    "compute_metrics",
]
