"""
Prefix-tuning Core Implementation

This module contains the core Prefix-tuning implementation including:
- PrefixTuningConfig: Configuration class for prefix-tuning parameters
- PrefixTuningModel: Main model wrapper that injects prefix vectors into transformer layers

Prefix-tuning is a parameter-efficient fine-tuning method that:
1. Freezes all pretrained model parameters
2. Adds trainable prefix vectors to each transformer layer
3. Only trains the prefix parameters (typically 0.1-1% of total parameters)

Author: AI Assistant
Date: 2026-03-16
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    BartForConditionalGeneration,
    PreTrainedModel,
)


@dataclass
class PrefixTuningConfig:
    """
    Configuration class for Prefix-tuning.
    
    Args:
        model_name_or_path: Pretrained model name or path (e.g., 'gpt2', 'facebook/bart-base')
        model_type: Type of model ('causal' for GPT-2, 'seq2seq' for BART)
        prefix_length: Number of prefix tokens to add (typically 10-100)
        num_layers: Number of transformer layers to add prefix to (usually all layers)
        hidden_size: Hidden size of the prefix vectors (often same as model's hidden size)
        bottleneck_size: Size of the bottleneck in reparameterization (typically 512)
        dropout: Dropout rate for prefix vectors
        prefix_projection: Whether to use projection layer for prefix (reparameterization trick)
        device: Device to run the model on
    """
    model_name_or_path: str = "gpt2"
    model_type: str = "causal"  # 'causal' or 'seq2seq'
    prefix_length: int = 20
    num_layers: Optional[int] = None  # If None, use all layers from model
    hidden_size: Optional[int] = None  # If None, infer from model
    bottleneck_size: int = 512
    dropout: float = 0.1
    prefix_projection: bool = True  # Use reparameterization trick
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.prefix_length <= 0:
            raise ValueError("prefix_length must be positive")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.model_type not in ["causal", "seq2seq"]:
            raise ValueError("model_type must be 'causal' or 'seq2seq'")


class PrefixTuningModel(nn.Module):
    """
    Prefix-tuning wrapper for pretrained transformer models.
    
    This class wraps a pretrained model (GPT-2 or BART) and adds trainable
    prefix vectors to each transformer layer. The key features are:
    
    1. Parameter Freezing: All original model parameters are frozen
    2. Prefix Injection: Trainable prefix vectors are injected into attention layers
    3. Multi-layer Support: Prefix can be added to multiple transformer layers
    4. Reparameterization: Uses a bottleneck MLP for better optimization
    
    The prefix vectors are concatenated with the input embeddings and
    processed together through the transformer layers.
    
    Example:
        >>> config = PrefixTuningConfig(model_name_or_path="gpt2", prefix_length=20)
        >>> model = PrefixTuningModel(config)
        >>> outputs = model.generate(input_ids, attention_mask)
    """
    
    def __init__(self, config: PrefixTuningConfig):
        """
        Initialize the Prefix-tuning model.
        
        Args:
            config: PrefixTuningConfig object with model parameters
        """
        super().__init__()
        self.config = config
        
        # Load the base pretrained model
        self._load_base_model()
        
        # Infer hidden size from model if not specified
        if config.hidden_size is None:
            self.config.hidden_size = self._get_hidden_size()
        
        # Infer number of layers from model if not specified
        if config.num_layers is None:
            self.config.num_layers = self._get_num_layers()
        
        # Create prefix parameters
        self._create_prefix_parameters()
        
        # Freeze base model parameters (only train prefix)
        self._freeze_base_model()
        
        # Move to device
        self.to(config.device)
    
    def _load_base_model(self):
        """Load the base pretrained model based on model type."""
        if self.config.model_type == "causal":
            # For causal language models (GPT-2)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path
            )
            self.model_type = "causal"
        elif self.config.model_type == "seq2seq":
            # For sequence-to-sequence models (BART)
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name_or_path
            )
            self.model_type = "seq2seq"
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        print(f"Loaded {self.config.model_name_or_path} ({self.model_type} model)")
    
    def _get_hidden_size(self) -> int:
        """Get the hidden size from the base model."""
        if hasattr(self.base_model.config, "hidden_size"):
            return self.base_model.config.hidden_size
        elif hasattr(self.base_model.config, "d_model"):
            return self.base_model.config.d_model
        else:
            # Default fallback
            return 768
    
    def _get_num_layers(self) -> int:
        """Get the number of transformer layers from the base model."""
        if hasattr(self.base_model.config, "num_hidden_layers"):
            return self.base_model.config.num_hidden_layers
        elif hasattr(self.base_model.config, "num_layers"):
            return self.base_model.config.num_layers
        else:
            # Default fallback
            return 12
    
    def _create_prefix_parameters(self):
        """
        Create trainable prefix parameters with reparameterization.
        
        Uses the reparameterization trick from the original Prefix-tuning paper:
        - Instead of directly optimizing prefix vectors, optimize through a bottleneck MLP
        - This improves optimization stability and performance
        
        The prefix has shape: (num_layers, 2, prefix_length, hidden_size)
        where 2 represents key and value prefixes for attention.
        """
        # Prefix embeddings: [num_layers, 2, prefix_length, hidden_size]
        # The '2' dimension is for key and value prefixes in attention
        self.prefix_tokens = nn.Parameter(
            torch.randn(
                self.config.num_layers,
                2,  # key and value
                self.config.prefix_length,
                self.config.hidden_size
            )
        )
        
        # Reparameterization: bottleneck MLP for better optimization
        if self.config.prefix_projection:
            self.prefix_proj = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.bottleneck_size),
                nn.Tanh(),
                nn.Linear(self.config.bottleneck_size, self.config.hidden_size),
                nn.Dropout(self.config.dropout)
            )
        else:
            self.prefix_proj = None
        
        # Dropout for prefix
        self.prefix_dropout = nn.Dropout(self.config.dropout)
        
        print(f"Created prefix parameters:")
        print(f"  - Layers: {self.config.num_layers}")
        print(f"  - Prefix length: {self.config.prefix_length}")
        print(f"  - Hidden size: {self.config.hidden_size}")
        print(f"  - Bottleneck size: {self.config.bottleneck_size}")
        print(f"  - Trainable parameters: {self.count_trainable_parameters():,}")
    
    def _freeze_base_model(self):
        """Freeze all parameters in the base model (only train prefix)."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Ensure prefix parameters are trainable
        self.prefix_tokens.requires_grad = True
        if self.prefix_proj is not None:
            for param in self.prefix_proj.parameters():
                param.requires_grad = True
    
    def count_trainable_parameters(self) -> int:
        """Count the number of trainable parameters (prefix only)."""
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
    
    def count_total_parameters(self) -> int:
        """Count total parameters (base model + prefix)."""
        return sum(p.numel() for p in self.parameters())
    
    def get_prefix(self) -> torch.Tensor:
        """
        Get the processed prefix vectors after reparameterization.
        
        Returns:
            Prefix tensor of shape (num_layers, 2, prefix_length, hidden_size)
        """
        if self.prefix_proj is not None:
            # Apply reparameterization
            prefix = self.prefix_proj(self.prefix_tokens)
        else:
            prefix = self.prefix_tokens
        
        return self.prefix_dropout(prefix)
    
    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Prepare attention mask that accounts for prefix tokens.
        
        The attention mask needs to be extended to include prefix tokens.
        Prefix tokens should always be attended to (mask = 1).
        
        Args:
            attention_mask: Original attention mask (batch_size, seq_length)
            batch_size: Batch size
            
        Returns:
            Extended attention mask including prefix tokens
        """
        # Create mask for prefix tokens (always attend to prefix)
        prefix_mask = torch.ones(
            batch_size,
            self.config.prefix_length,
            device=self.config.device
        )
        
        # Concatenate prefix mask with original mask
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        return extended_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the prefix-tuning model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            labels: Optional labels for training (batch_size, seq_length)
            **kwargs: Additional arguments passed to base model
            
        Returns:
            Dictionary containing:
                - loss: Training loss (if labels provided)
                - logits: Model outputs
                - past_key_values: Prefix key/values for generation
        """
        batch_size = input_ids.shape[0]
        
        # Get prefix vectors
        prefix = self.get_prefix()  # (num_layers, 2, prefix_length, hidden_size)
        
        # Split prefix into key and value components
        # Shape: (num_layers, prefix_length, hidden_size) each
        prefix_keys = prefix[:, 0]  # (num_layers, prefix_length, hidden_size)
        prefix_values = prefix[:, 1]  # (num_layers, prefix_length, hidden_size)
        
        # Prepare attention mask with prefix
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        extended_attention_mask = self._prepare_attention_mask(
            attention_mask, batch_size
        )
        
        # Get input embeddings
        if self.model_type == "causal":
            # For GPT-2 style models
            inputs_embeds = self.base_model.transformer.wte(input_ids)
        else:
            # For BART style models
            inputs_embeds = self.base_model.model.shared(input_ids)
        
        # Concatenate prefix with input embeddings
        # Expand prefix to match batch size
        prefix_keys_expanded = prefix_keys.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )  # (batch_size, num_layers, prefix_length, hidden_size)
        prefix_values_expanded = prefix_values.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )
        
        # For causal models, we need to handle the prefix injection differently
        # We'll use the past_key_values mechanism
        if self.model_type == "causal":
            # Reshape prefix for past_key_values format
            # GPT-2 expects: tuple of (batch_size, num_heads, seq_length, head_dim)
            past_key_values = self._format_prefix_for_causal(
                prefix_keys, prefix_values, batch_size
            )
            
            # Forward through base model with prefix
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=extended_attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                **kwargs
            )
        else:
            # For seq2seq models (BART), handle encoder-decoder attention
            past_key_values = self._format_prefix_for_seq2seq(
                prefix_keys, prefix_values, batch_size
            )
            
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=extended_attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                **kwargs
            )
        
        return outputs
    
    def _format_prefix_for_causal(
        self,
        prefix_keys: torch.Tensor,
        prefix_values: torch.Tensor,
        batch_size: int
    ) -> Tuple:
        """
        Format prefix for causal language models (GPT-2).
        
        Args:
            prefix_keys: Prefix keys (num_layers, prefix_length, hidden_size)
            prefix_values: Prefix values (num_layers, prefix_length, hidden_size)
            batch_size: Batch size
            
        Returns:
            Tuple of past_key_values in format expected by GPT-2
        """
        # Reshape for multi-head attention
        # (num_layers, batch_size, num_heads, prefix_length, head_dim)
        num_heads = self.base_model.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads
        
        # Reshape keys and values
        prefix_keys = prefix_keys.view(
            self.config.num_layers,
            batch_size,
            self.config.prefix_length,
            num_heads,
            head_dim
        ).permute(0, 1, 3, 2, 4)  # (num_layers, batch_size, num_heads, prefix_length, head_dim)
        
        prefix_values = prefix_values.view(
            self.config.num_layers,
            batch_size,
            self.config.prefix_length,
            num_heads,
            head_dim
        ).permute(0, 1, 3, 2, 4)
        
        # Create tuple format expected by transformers
        past_key_values = tuple(
            (prefix_keys[i], prefix_values[i])
            for i in range(self.config.num_layers)
        )
        
        return past_key_values
    
    def _format_prefix_for_seq2seq(
        self,
        prefix_keys: torch.Tensor,
        prefix_values: torch.Tensor,
        batch_size: int
    ) -> Tuple:
        """
        Format prefix for sequence-to-sequence models (BART).
        
        Args:
            prefix_keys: Prefix keys (num_layers, prefix_length, hidden_size)
            prefix_values: Prefix values (num_layers, prefix_length, hidden_size)
            batch_size: Batch size
            
        Returns:
            Tuple of past_key_values in format expected by BART
        """
        # Similar to causal but adapted for encoder-decoder architecture
        num_heads = self.base_model.config.encoder_attention_heads
        head_dim = self.config.hidden_size // num_heads
        
        # Reshape keys and values
        prefix_keys = prefix_keys.view(
            self.config.num_layers,
            batch_size,
            self.config.prefix_length,
            num_heads,
            head_dim
        ).permute(0, 1, 3, 2, 4)
        
        prefix_values = prefix_values.view(
            self.config.num_layers,
            batch_size,
            self.config.prefix_length,
            num_heads,
            head_dim
        ).permute(0, 1, 3, 2, 4)
        
        # Create tuple format for both encoder and decoder
        past_key_values = tuple(
            (prefix_keys[i], prefix_values[i])
            for i in range(self.config.num_layers)
        )
        
        return past_key_values
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        max_new_tokens: int = 50,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the prefix-tuning model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            max_length: Maximum total length of generated sequence
            max_new_tokens: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        
        # Get prefix vectors
        prefix = self.get_prefix()
        prefix_keys = prefix[:, 0]
        prefix_values = prefix[:, 1]
        
        # Format prefix for model type
        if self.model_type == "causal":
            past_key_values = self._format_prefix_for_causal(
                prefix_keys, prefix_values, batch_size
            )
        else:
            past_key_values = self._format_prefix_for_seq2seq(
                prefix_keys, prefix_values, batch_size
            )
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        extended_attention_mask = self._prepare_attention_mask(
            attention_mask, batch_size
        )
        
        # Set max_length if not provided
        if max_length is None:
            max_length = input_ids.shape[1] + max_new_tokens
        
        # Generate using base model's generate method
        outputs = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=extended_attention_mask,
            past_key_values=past_key_values,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )
        
        return outputs
    
    def save_prefix(self, save_path: str):
        """
        Save only the prefix parameters to a file.
        
        Args:
            save_path: Path to save the prefix parameters
        """
        prefix_state = {
            "prefix_tokens": self.prefix_tokens.cpu().detach(),
            "config": self.config,
        }
        
        if self.prefix_proj is not None:
            prefix_state["prefix_proj"] = self.prefix_proj.cpu().state_dict()
        
        torch.save(prefix_state, save_path)
        print(f"Prefix saved to {save_path}")
    
    def load_prefix(self, load_path: str):
        """
        Load prefix parameters from a file.
        
        Args:
            load_path: Path to load the prefix parameters from
        """
        prefix_state = torch.load(load_path, map_location=self.config.device)
        
        self.prefix_tokens.data = prefix_state["prefix_tokens"].to(self.config.device)
        
        if "prefix_proj" in prefix_state and self.prefix_proj is not None:
            self.prefix_proj.load_state_dict(prefix_state["prefix_proj"])
        
        print(f"Prefix loaded from {load_path}")
    
    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        """Get all trainable parameters (prefix only)."""
        params = {"prefix_tokens": self.prefix_tokens}
        if self.prefix_proj is not None:
            for name, param in self.prefix_proj.named_parameters():
                params[f"prefix_proj.{name}"] = param
        return params


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Testing Prefix-tuning Implementation")
    print("=" * 60)
    
    # Test with GPT-2
    print("\n1. Testing with GPT-2 (causal model)...")
    config_gpt2 = PrefixTuningConfig(
        model_name_or_path="gpt2",
        model_type="causal",
        prefix_length=10,
        prefix_projection=True
    )
    model_gpt2 = PrefixTuningModel(config_gpt2)
    
    print(f"Total parameters: {model_gpt2.count_total_parameters():,}")
    print(f"Trainable parameters: {model_gpt2.count_trainable_parameters():,}")
    print(f"Trainable %: {100 * model_gpt2.count_trainable_parameters() / model_gpt2.count_total_parameters():.3f}%")
    
    # Test with dummy input
    dummy_input = torch.randint(0, 1000, (2, 20)).to(config_gpt2.device)
    dummy_mask = torch.ones_like(dummy_input)
    
    with torch.no_grad():
        outputs = model_gpt2(dummy_input, attention_mask=dummy_mask)
        print(f"Output logits shape: {outputs.logits.shape}")
    
    # Test generation
    generated = model_gpt2.generate(dummy_input, max_new_tokens=10)
    print(f"Generated shape: {generated.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
