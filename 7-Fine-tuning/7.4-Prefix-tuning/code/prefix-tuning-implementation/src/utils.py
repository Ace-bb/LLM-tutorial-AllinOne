"""
Utility Functions for Prefix-tuning

This module provides utility functions for:
- Model and tokenizer loading
- Prefix saving and loading
- Random seed setting for reproducibility
- Metrics computation
- Data preprocessing

Author: AI Assistant
Date: 2026-03-16
"""

import torch
import random
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
import os


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    This function sets the seed for all random number generators used
    in the project to ensure reproducible results.
    
    Args:
        seed: Random seed value (default: 42)
    
    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def load_model_and_tokenizer(
    model_name_or_path: str,
    model_type: str = "causal",
    cache_dir: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load pretrained model and tokenizer.
    
    Args:
        model_name_or_path: Name or path of pretrained model
        model_type: Type of model ('causal' for GPT-2, 'seq2seq' for BART)
        cache_dir: Directory to cache model files
    
    Returns:
        Tuple of (model, tokenizer)
    
    Example:
        >>> model, tokenizer = load_model_and_tokenizer("gpt2", "causal")
    """
    print(f"Loading model: {model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
    
    # Load model based on type
    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )
    elif model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Model loaded successfully")
    print(f"  - Model type: {model_type}")
    print(f"  - Parameters: {model.num_parameters():,}")
    
    return model, tokenizer


def save_prefix(
    prefix_state: Dict[str, torch.Tensor],
    save_path: str,
    config: Optional[Dict] = None,
):
    """
    Save prefix parameters to file.
    
    Args:
        prefix_state: Dictionary containing prefix tensors
        save_path: Path to save the file
        config: Optional configuration dictionary to save
    
    Example:
        >>> save_prefix({"prefix_tokens": tokens}, "prefix.pt", config)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save tensors
    torch.save(prefix_state, save_path)
    
    # Save config if provided
    if config is not None:
        config_path = save_path.replace(".pt", "_config.json")
        import json
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    print(f"Prefix saved to {save_path}")


def load_prefix(
    load_path: str,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load prefix parameters from file.
    
    Args:
        load_path: Path to load the file from
        device: Device to load tensors to
    
    Returns:
        Dictionary containing prefix tensors
    
    Example:
        >>> prefix_state = load_prefix("prefix.pt", "cuda")
    """
    prefix_state = torch.load(load_path, map_location=device)
    print(f"Prefix loaded from {load_path}")
    return prefix_state


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for sequence generation.
    
    Args:
        predictions: Predicted token IDs
        labels: Ground truth token IDs
        ignore_index: Token ID to ignore in metrics (e.g., padding)
    
    Returns:
        Dictionary of metrics including:
            - accuracy: Token-level accuracy
            - exact_match: Exact match rate
    
    Example:
        >>> metrics = compute_metrics(preds, labels)
    """
    # Flatten predictions and labels
    predictions = predictions.view(-1)
    labels = labels.view(-1)
    
    # Create mask for valid tokens
    mask = labels != ignore_index
    
    # Compute accuracy
    correct = (predictions == labels) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    
    # Compute exact match (per sequence)
    predictions_2d = predictions.view(labels.shape[0], -1)
    labels_2d = labels.view(labels.shape[0], -1)
    mask_2d = labels_2d != ignore_index
    
    exact_matches = []
    for pred, label, m in zip(predictions_2d, labels_2d, mask_2d):
        pred_valid = pred[m]
        label_valid = label[m]
        if len(pred_valid) == len(label_valid):
            exact_matches.append(torch.all(pred_valid == label_valid).item())
        else:
            exact_matches.append(False)
    
    exact_match = sum(exact_matches) / len(exact_matches)
    
    return {
        "accuracy": accuracy,
        "exact_match": exact_match,
    }


def tokenize_batch(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a batch of texts.
    
    Args:
        texts: List of input texts
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate long sequences
        return_tensors: Type of tensors to return
    
    Returns:
        Tokenized batch dictionary
    
    Example:
        >>> batch = tokenize_batch(texts, tokenizer, max_length=128)
    """
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    
    return encodings


def prepare_training_data(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> List[Dict[str, torch.Tensor]]:
    """
    Prepare training data from texts.
    
    This function tokenizes texts and prepares them for training by
    creating input_ids, attention_mask, and labels.
    
    Args:
        texts: List of input texts
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
    
    Returns:
        List of dictionaries containing tokenized data
    
    Example:
        >>> data = prepare_training_data(texts, tokenizer)
    """
    data = []
    
    for text in texts:
        # Tokenize
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Create data dictionary
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
        }
        
        data.append(item)
    
    return data


def create_prefix_dataset(
    input_texts: List[str],
    target_texts: Optional[List[str]] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: int = 512,
    model_type: str = "causal",
) -> List[Dict[str, torch.Tensor]]:
    """
    Create a dataset for prefix-tuning training.
    
    Args:
        input_texts: List of input texts (prompts)
        target_texts: List of target texts (for seq2seq models)
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        model_type: Type of model ('causal' or 'seq2seq')
    
    Returns:
        List of dictionaries ready for training
    
    Example:
        >>> dataset = create_prefix_dataset(inputs, targets, tokenizer)
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    dataset = []
    
    if model_type == "causal":
        # For causal models, concatenate input and target
        for i, text in enumerate(input_texts):
            # Tokenize full sequence
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            
            # Labels are the same as input_ids for causal LM
            labels = input_ids.clone()
            
            # Mask the loss for input portion (optional)
            # Only compute loss on target portion
            
            dataset.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
    
    elif model_type == "seq2seq":
        # For seq2seq models, separate input and target
        if target_texts is None:
            raise ValueError("target_texts required for seq2seq models")
        
        for input_text, target_text in zip(input_texts, target_texts):
            # Tokenize input
            input_encoding = tokenizer(
                input_text,
                max_length=max_length // 2,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            # Tokenize target
            target_encoding = tokenizer(
                target_text,
                max_length=max_length // 2,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            dataset.append({
                "input_ids": input_encoding["input_ids"].squeeze(0),
                "attention_mask": input_encoding["attention_mask"].squeeze(0),
                "labels": target_encoding["input_ids"].squeeze(0),
            })
    
    return dataset


class PrefixDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for prefix-tuning.
    
    This dataset class wraps tokenized data for efficient training.
    
    Args:
        data: List of dictionaries containing tokenized data
    
    Example:
        >>> dataset = PrefixDataset(data)
        >>> loader = DataLoader(dataset, batch_size=4)
    """
    
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    
    Example:
        >>> format_time(3665)
        '1h 1m 5s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_model_size(model: torch.nn.Module) -> str:
    """
    Get model size in human-readable format.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size string (e.g., "125M", "1.2B")
    
    Example:
        >>> get_model_size(model)
        '125M'
    """
    num_params = model.num_parameters()
    
    if num_params >= 1e9:
        return f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.1f}K"
    else:
        return str(num_params)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test set_seed
    set_seed(42)
    
    # Test format_time
    print(f"3665 seconds = {format_time(3665)}")
    
    print("All utility tests passed!")
