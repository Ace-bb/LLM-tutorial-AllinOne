"""
Prefix-tuning Trainer

This module provides the training logic for prefix-tuning, including:
- PrefixTrainer: Main trainer class with complete training loop
- Training utilities: gradient clipping, learning rate scheduling, etc.

The trainer implements:
1. Efficient training (only prefix parameters are updated)
2. Gradient accumulation for larger effective batch sizes
3. Learning rate scheduling with warmup
4. Evaluation and checkpointing
5. Mixed precision training support

Author: AI Assistant
Date: 2026-03-16
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from tqdm import tqdm
import os
from dataclasses import dataclass
import json


@dataclass
class TrainingArguments:
    """
    Training arguments for prefix-tuning.
    
    Args:
        output_dir: Directory to save checkpoints and outputs
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Steps to accumulate gradients before updating
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        warmup_ratio: Ratio of warmup steps over total training steps
        max_grad_norm: Maximum gradient norm for clipping
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        save_total_limit: Maximum number of checkpoints to keep
        fp16: Whether to use mixed precision training
        seed: Random seed for reproducibility
    """
    output_dir: str = "./output"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = False
    seed: int = 42


class PrefixTrainer:
    """
    Trainer class for prefix-tuning.
    
    This class handles the complete training loop including:
    - Training and evaluation
    - Checkpointing and resumption
    - Learning rate scheduling
    - Gradient accumulation
    - Mixed precision training
    
    Example:
        >>> trainer = PrefixTrainer(model, train_dataset, eval_dataset, args)
        >>> trainer.train()
        >>> trainer.save_model()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PrefixTuningModel to train
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            args: Training arguments
            data_collator: Function to collate batch data
            compute_metrics: Function to compute evaluation metrics
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args or TrainingArguments()
        self.data_collator = data_collator or self._default_data_collator
        self.compute_metrics = compute_metrics
        
        # Set device
        self.device = self.model.config.device
        
        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Setup training components
        self._setup_training()
    
    def _default_data_collator(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Default data collator for batching.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Batched tensor dictionary
        """
        batch = {}
        for key in features[0].keys():
            if isinstance(features[0][key], torch.Tensor):
                batch[key] = torch.stack([f[key] for f in features])
            else:
                batch[key] = torch.tensor([f[key] for f in features])
        return batch
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and other training components."""
        # Get only trainable parameters (prefix)
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        
        # Create optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        
        # Calculate total training steps
        total_steps = (
            len(self.train_dataset)
            // (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps)
            // self.args.num_train_epochs
        )
        
        # Create learning rate scheduler with warmup
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Setup mixed precision if enabled
        if self.args.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        print(f"Training setup complete:")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Warmup steps: {warmup_steps}")
        print(f"  - Mixed precision: {self.args.fp16}")
    
    def _compute_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss for a batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss tensor
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Get loss
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        return loss
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, float]:
        """
        Main training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training metrics dictionary
        """
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )
        
        # Training metrics
        global_step = 0
        best_eval_loss = float('inf')
        training_metrics = {
            "total_loss": 0.0,
            "best_eval_loss": float('inf'),
            "global_step": 0,
        }
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
            print(f"Resumed from checkpoint: {resume_from_checkpoint}")
        
        print(f"\nStarting training for {self.args.num_train_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Trainable parameters: {self.model.count_trainable_parameters():,}")
        
        # Training loop
        for epoch in range(start_epoch, self.args.num_train_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.args.num_train_epochs}")
            print(f"{'='*60}")
            
            # Set model to training mode
            self.model.train()
            
            # Progress bar for epoch
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}",
                leave=False
            )
            
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(progress_bar):
                # Forward and backward pass
                loss = self._training_step(batch)
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Logging
                if global_step % self.args.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"\nStep {global_step}: loss={avg_loss:.4f}, lr={lr:.6f}")
                
                # Evaluation
                if self.eval_dataset and global_step % self.args.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    print(f"Evaluation at step {global_step}: {eval_metrics}")
                    
                    # Save best model
                    if eval_metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        self._save_checkpoint(global_step, is_best=True)
                
                # Save checkpoint
                if global_step % self.args.save_steps == 0:
                    self._save_checkpoint(global_step)
                
                # Update learning rate
                self.scheduler.step()
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            print(f"\nEpoch {epoch + 1} summary:")
            print(f"  - Average loss: {avg_epoch_loss:.4f}")
            print(f"  - Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            training_metrics["total_loss"] += avg_epoch_loss
        
        # Final evaluation
        if self.eval_dataset:
            print("\nFinal evaluation...")
            final_metrics = self.evaluate()
            training_metrics.update(final_metrics)
        
        # Save final model
        self._save_checkpoint(global_step, is_final=True)
        
        training_metrics["global_step"] = global_step
        training_metrics["best_eval_loss"] = best_eval_loss
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  - Total steps: {global_step}")
        print(f"  - Best eval loss: {best_eval_loss:.4f}")
        print(f"{'='*60}")
        
        return training_metrics
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss tensor
        """
        # Mixed precision training
        if self.args.fp16 and self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss = self._compute_loss(batch)
                loss = loss / self.args.gradient_accumulation_steps
            
            # Scale and backward
            self.scaler.scale(loss).backward()
            
            # Optimizer step with gradient clipping
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            # Standard training
            loss = self._compute_loss(batch)
            loss = loss / self.args.gradient_accumulation_steps
            
            loss.backward()
            
            # Gradient accumulation and update
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return loss.detach()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.eval_dataset:
            return {}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create data loader
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        
        # Evaluation metrics
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        # Progress bar
        progress_bar = tqdm(eval_loader, desc="Evaluating")
        
        for batch in progress_bar:
            # Compute loss
            loss = self._compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            outputs = self.model(**batch)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.append(predictions.cpu())
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu())
        
        # Compute metrics
        avg_loss = total_loss / max(num_batches, 1)
        
        metrics = {
            "eval_loss": avg_loss,
        }
        
        # Custom metrics if provided
        if self.compute_metrics and all_predictions and all_labels:
            custom_metrics = self.compute_metrics(
                torch.cat(all_predictions),
                torch.cat(all_labels)
            )
            metrics.update(custom_metrics)
        
        # Set back to training mode
        self.model.train()
        
        return metrics
    
    def _save_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        is_final: bool = False
    ):
        """
        Save a training checkpoint.
        
        Args:
            global_step: Current global step
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint
        """
        checkpoint_dir = os.path.join(
            self.args.output_dir,
            f"checkpoint-{global_step}"
        )
        
        if is_best:
            checkpoint_dir = os.path.join(self.args.output_dir, "best_model")
        elif is_final:
            checkpoint_dir = os.path.join(self.args.output_dir, "final_model")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save prefix parameters
        prefix_path = os.path.join(checkpoint_dir, "prefix.pt")
        self.model.save_prefix(prefix_path)
        
        # Save training state
        state = {
            "global_step": global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "args": vars(self.args),
        }
        
        if self.scaler is not None:
            state["scaler_state"] = self.scaler.state_dict()
        
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        torch.save(state, state_path)
        
        # Save config
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(self.model.config), f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
        
        # Clean up old checkpoints
        if not is_best and not is_final:
            self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = []
        for item in os.listdir(self.args.output_dir):
            if item.startswith("checkpoint-"):
                checkpoints.append(item)
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        
        # Remove old checkpoints
        while len(checkpoints) > self.args.save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            old_path = os.path.join(self.args.output_dir, old_checkpoint)
            
            # Remove directory
            import shutil
            shutil.rmtree(old_path)
            print(f"Removed old checkpoint: {old_checkpoint}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # Load prefix parameters
        prefix_path = os.path.join(checkpoint_path, "prefix.pt")
        self.model.load_prefix(prefix_path)
        
        # Load training state
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        state = torch.load(state_path, map_location=self.device)
        
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        
        if self.scaler is not None and "scaler_state" in state:
            self.scaler.load_state_dict(state["scaler_state"])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_model(self, save_path: str):
        """
        Save the final model.
        
        Args:
            save_path: Path to save the model
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save prefix
        prefix_path = os.path.join(save_path, "prefix.pt")
        self.model.save_prefix(prefix_path)
        
        # Save config
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(self.model.config), f, indent=2)
        
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("PrefixTrainer example")
    print("See scripts/train.py for complete training example")
