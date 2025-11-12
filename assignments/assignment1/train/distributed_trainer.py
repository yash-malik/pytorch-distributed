import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Any, Optional, Union
from pathlib import Path

from train.trainer import Trainer


class DistributedTrainer(Trainer):
    """
    Distributed version of Trainer for DDP/FSDP training.
    
    Key features:
    - no_sync context for gradient accumulation (avoids unnecessary all-reduce)
    - Rank-aware logging and checkpointing (only rank 0)
    - Global loss aggregation across all ranks
    
    Args:
        model: nn.Module (should be wrapped with DDP/FSDP before passing)
        optimizer: torch.optim.Optimizer
        lr_scheduler: Optional learning rate scheduler
        max_steps: Maximum training steps
        global_batch_size: Total batch size across all ranks
        micro_batch_size: Batch size per rank (local_batch_size)
        log_every_n_steps: Logging frequency
        save_every_n_steps: Checkpointing frequency
        checkpoint_dir: Directory for checkpoints
        ddp_enabled: Explicit flag to enable DDP/FSDP features
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        max_steps: int = 1000,
        global_batch_size: int = 32,
        micro_batch_size: int = 8,
        log_every_n_steps: int = 10,
        save_every_n_steps: Optional[int] = None,
        checkpoint_dir: str = "checkpoints",
        ddp_enabled: bool = True
    ):
        # Initialize parent
        super().__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            log_every_n_steps=log_every_n_steps,
            save_every_n_steps=save_every_n_steps,
            checkpoint_dir=checkpoint_dir
        )
        
        # Distributed settings
        self.ddp_enabled = ddp_enabled
        
        if ddp_enabled:
            # Auto-detect rank and world_size
            self.rank = int(os.environ.get('RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            
            # Verify distributed is initialized
            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed is not initialized. "
                    "Please call torch.distributed.init_process_group() before creating DistributedTrainer."
                )
            
            # Check if model supports no_sync (DDP/FSDP)
            if not hasattr(model, 'no_sync'):
                raise RuntimeError(
                    "Model does not have 'no_sync' method. "
                    "Please wrap your model with DDP or FSDP before passing to DistributedTrainer."
                )
        else:
            self.rank = 0
            self.world_size = 1
        
        # TODO Task 1: Recalculate gradient accumulation considering world_size
        # Key insight: With multiple ranks, each step processes micro_batch_size * world_size sequences
        # Example: global=32, micro=8, world=2 -> effective_per_step=16, grad_acc=32/16=2
        effective_batch_per_step = micro_batch_size * self.world_size
        self.grad_accumulation_steps = 0  # global_batch_size // effective_batch_per_step
        
        print(f"DistributedTrainer initialized: rank={self.rank}, world_size={self.world_size}, "
              f"grad_acc_steps={self.grad_accumulation_steps}, ddp_enabled={ddp_enabled}")
    
    def training_step(self, inputs: torch.Tensor, targets: torch.Tensor, should_sync: bool) -> torch.Tensor:
        """
        Training step with DDP/FSDP no_sync support.
        
        Key concept: For gradient accumulation, we only want to sync gradients
        on the last micro-batch to avoid unnecessary communication overhead.
        
        Args:
            inputs: Input tokens
            targets: Target tokens
            should_sync: Whether to sync gradients (True for last micro-batch)
        
        Returns:
            Loss value
        """
        # Forward pass
        logits = self.model(inputs)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1)
        )
        
        # TODO 2: Implement backward pass with conditional gradient synchronization
        # Hint: Use model.no_sync() context when should_sync=False
        # This prevents all_reduce communication until the last micro-batch
        scaled_loss = loss / self.grad_accumulation_steps
        
        if self.ddp_enabled and not should_sync:
            # Don't sync gradients for intermediate micro-batches
            # with self.model.no_sync():
            #     scaled_loss.backward()
            pass
        else:
            # Sync gradients on last micro-batch (triggers all_reduce in DDP)
            scaled_loss.backward()
        
        return loss
    
    def _aggregate_loss(self, loss: float, device: torch.device) -> float:
        """
        Aggregate loss across all ranks for accurate logging.
        
        Key concept: Each rank computes loss on its local batch. To get the
        true global loss, we need to average losses across all ranks using
        all_reduce collective operation.
        
        Args:
            loss: Local loss value (this rank's batch only)
            device: Device for tensor operations
        
        Returns:
            Global averaged loss (across all ranks)
        """
        if not self.ddp_enabled or self.world_size == 1:
            return loss
        
        # TODO 3: Convert loss to tensor and perform all_reduce averaging
        # Hint: Use dist.all_reduce() with ReduceOp.AVG
        loss_tensor = torch.tensor([loss], device=device)
        # dist.all_reduce(tensor=, op=)

        return loss_tensor.item()
    
    def train(self, dataloader, profiler: Optional[Any] = None) -> None:
        """Main training loop with distributed support."""
        self.model.train()
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if self.start_time:
            self.start_time.record()
        
        if self.rank == 0:
            print(f"Starting distributed training for {self.max_steps} steps")
        
        for batch in dataloader:
            if self.current_step >= self.max_steps:
                break
            
            # Unpack batch
            inputs, targets = batch
            device = inputs.device
            
            # Determine if this is the last micro-batch (should sync gradients)
            is_last_microbatch = (self.batch_count + 1) % self.grad_accumulation_steps == 0
            
            # Execute training step
            loss = self.training_step(inputs, targets, should_sync=is_last_microbatch)
            self.loss_accumulator += loss.item()
            
            # Increment batch counter
            self.batch_count += 1
            should_step = self.batch_count % self.grad_accumulation_steps == 0
            
            # Optimizer step after gradient accumulation
            if should_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Logging
                if self.current_step % self.log_every_n_steps == 0:
                    # Aggregate loss across all ranks (ALL ranks must participate)
                    avg_loss = self.loss_accumulator / self.grad_accumulation_steps
                    global_loss = self._aggregate_loss(avg_loss, device)
                    
                    # Only rank 0 prints
                    if self.rank == 0:
                        lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.optimizer.param_groups[0]['lr']
                        
                        if end_time:
                            end_time.record()
                            torch.cuda.synchronize()
                            elapsed = self.start_time.elapsed_time(end_time) / 1000.0
                        else:
                            elapsed = 0.0
                        
                        print(f"step={self.current_step} | loss={global_loss:.4f} | lr={lr:.2e} | time={elapsed:.1f}s")
                
                # All ranks have the same model state due to gradient synchronization
                if (self.save_every_n_steps is not None and 
                    self.current_step > 0 and 
                    self.current_step % self.save_every_n_steps == 0 and
                    self.rank == 0):
                    checkpoint_path = f"{self.checkpoint_dir}/checkpoint_step_{self.current_step}.pt"
                    self.save_checkpoint(checkpoint_path)
                    print(f"Saved: {checkpoint_path}")
                
                self.loss_accumulator = 0.0
                self.current_step += 1
            
            # Step profiler
            if profiler is not None:
                profiler.step()
        
        if self.rank == 0:
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                elapsed = self.start_time.elapsed_time(end_time) / 1000.0
            else:
                elapsed = 0.0
            print(f"Training completed in {elapsed:.1f}s")
