"""
Baseline Single GPU Training

This serves as a reference implementation for comparing with DDP and FSDP.
Run: uv run python train_baseline.py
"""

import torch
import torch.nn as nn
from transformers import AutoConfig
import os

from model.my_gpt2 import MyGPT2LMHeadModel
from data.data_loader import KJJ0DataLoader, download_fineweb10B_files
from train.trainer import Trainer


def main():
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Configuration
    config = AutoConfig.from_pretrained("gpt2-large")  # GPT-2 Large (774M parameters)
    
    # Global batch size is consistent across all setups
    global_batch_size = 32   # Total effective batch size
    micro_batch_size = 8     # Batch size that fits in memory
    sequence_length = 1024
    max_steps = 20
    learning_rate = 3e-4
    
    grad_acc_steps = global_batch_size // micro_batch_size
    print(f"Global batch size: {global_batch_size}")
    print(f"Micro batch size: {micro_batch_size}")
    print(f"Gradient accumulation steps: {grad_acc_steps}")
    print(f"Sequence length: {sequence_length}")
    print(f"Max steps: {max_steps}")
    
    # Model setup
    print("Creating model...")
    model = MyGPT2LMHeadModel(config)
    model.init_weights()
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data setup
    print("Setting up data...")
    files = download_fineweb10B_files(num_train_files=10)
    train_files = [f for f in files if "train" in str(f)]
    
    dataloader = KJJ0DataLoader(
        file_paths=train_files,
        batch_size=micro_batch_size,
        sequence_length=sequence_length,
        device=device
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=learning_rate * 0.1
    )
    
    # Trainer setup with gradient accumulation
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_steps=max_steps,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        log_every_n_steps=10
    )
    
    # Training with profiling
    print("Starting training with profiling...")
    from torch.profiler import profile, ProfilerActivity, schedule
    
    os.makedirs("outputs/traces/baseline", exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=lambda p: p.export_chrome_trace("outputs/traces/baseline/trace.json")
    ) as prof:
        trainer.train(dataloader, profiler=prof)
    
    print("Baseline training completed!")
    print(f"Traces saved to: outputs/traces/baseline/")


if __name__ == "__main__":
    main()
