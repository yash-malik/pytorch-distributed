"""
DDP Training Script

Train GPT-2 with DistributedDataParallel across multiple GPUs.

Run with torchrun:
    uv run torchrun --nproc_per_node=2 train_ddp.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig
import os

from model.my_gpt2 import MyGPT2LMHeadModel
from data.data_loader import download_fineweb10B_files
from data.distributed_data_loader import DistributedKJJ0DataLoader
from train.distributed_trainer import DistributedTrainer


def setup_distributed():    
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    
    if rank == 0:
        print(f"Initialized process group: world_size={world_size}, backend=nccl")
    
    return rank, world_size, device


def setup_ddp_model(model, device):
    """
    Wrap model with DistributedDataParallel.
    """
    # Move model to device first
    model = model.to(device)
    
    # TODO: Wrap with DDP
    local_rank = int(os.environ['LOCAL_RANK'])
    # Hint: DDP(model, device_ids=[local_rank])
    model = None
    
    return model


def main():
    # Setup distributed
    rank, world_size, device = setup_distributed()
    
    # Configuration - same as baseline!
    config = AutoConfig.from_pretrained("gpt2-large")  # GPT-2 Large (774M parameters)
    global_batch_size = 32
    micro_batch_size = 8  # Per rank
    sequence_length = 1024
    max_steps = 20
    learning_rate = 3e-4
    
    if rank == 0:
        print(f"Global batch size: {global_batch_size}")
        print(f"Micro batch size per rank: {micro_batch_size}")
        print(f"Effective batch size per rank: {micro_batch_size} (no grad accumulation with {world_size} GPUs)")
        print(f"Sequence length: {sequence_length}")
        print(f"Max steps: {max_steps}")
    
    # Key concept: All ranks must use the same seed for model initialization
    # to ensure identical starting weights across all processes
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Model setup
    if rank == 0:
        print("Creating model...")
    
    model = MyGPT2LMHeadModel(config)
    model.init_weights()
    
    # Wrap with DDP
    model = setup_ddp_model(model, device)
    
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data setup
    if rank == 0:
        print("Setting up distributed data...")
    
    files = download_fineweb10B_files(num_train_files=10)
    train_files = [f for f in files if "train" in str(f)]
    
    # Key concept: Each rank loads the same files but reads different token ranges
    dataloader = DistributedKJJ0DataLoader(
        file_paths=train_files,
        local_batch_size=micro_batch_size,
        sequence_length=sequence_length,
        device=device
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=learning_rate * 0.1
    )
    
    # Distributed trainer setup
    trainer = DistributedTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_steps=max_steps,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        log_every_n_steps=10,
        ddp_enabled=True
    )
    
    # Training with profiling (only rank 0 saves trace)
    if rank == 0:
        print("Starting DDP training with profiling...")
    
    from torch.profiler import profile, ProfilerActivity, schedule
    
    # Each rank generates its own trace
    os.makedirs("outputs/traces/ddp", exist_ok=True)
    trace_path = f"outputs/traces/ddp/rank{rank}_trace.json"
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=lambda p: p.export_chrome_trace(trace_path)
    ) as prof:
        trainer.train(dataloader, profiler=prof)
    
    if rank == 0:
        print("DDP training completed!")
        print(f"Traces saved to: outputs/traces/ddp/")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
