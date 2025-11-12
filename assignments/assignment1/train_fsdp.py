"""
FSDP Training Script

Train GPT-2 with FullyShardedDataParallel across multiple GPUs.
Uses the same training logic as DDP, with only the model wrapping changed.

Run with torchrun:
    uv run torchrun --nproc_per_node=2 train_fsdp.py --strategy FULL_SHARD
    uv run torchrun --nproc_per_node=2 train_fsdp.py --strategy SHARD_GRAD_OP
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from transformers import AutoConfig
import os
import argparse

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


def setup_fsdp_model(model, device, sharding_strategy_name='FULL_SHARD'):
    """
    Wrap model with FSDP using per-layer wrapping.
    
    Key concept: FSDP wraps each transformer block individually for
    fine-grained sharding control.
    
    Sharding strategies:
    - FULL_SHARD: Parameters, gradients, optimizer states all sharded
      -> Max memory savings, more communication (all_gather + reduce_scatter)
    - SHARD_GRAD_OP: Only gradients and optimizer states sharded, parameters replicated
      -> Moderate savings, less communication
    - NO_SHARD: Equivalent to DDP (for comparison)
    
    Args:
        model: Model to wrap
        device: Device to use
        sharding_strategy_name: Sharding strategy name
    """
    # Move model to device first
    model = model.to(device)
    
    strategy_map = {
        'FULL_SHARD': ShardingStrategy.FULL_SHARD,
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
        'NO_SHARD': ShardingStrategy.NO_SHARD
    }
    sharding_strategy = strategy_map[sharding_strategy_name]
    
    # TODO: Manually wrap each transformer block with FSDP
    # Key concept: Per-layer wrapping gives fine-grained control over sharding
    # We wrap each transformer.h[i] block individually, then wrap the full model
    
    # Hint: Loop through model.transformer.h and wrap each block
    # for i, block in enumerate(model.transformer.h):
    #     model.transformer.h[i] = FSDP(block, sharding_strategy=sharding_strategy)
    
    # TODO: Wrap the entire model with FSDP
    # Hint: model = FSDP(model, sharding_strategy=sharding_strategy)
    model = None  # Replace with manual FSDP wrapping
    
    return model


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='FULL_SHARD',
                       choices=['FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD'],
                       help='FSDP sharding strategy')
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, device = setup_distributed()
    
    # Configuration
    config = AutoConfig.from_pretrained("gpt2-large")  # GPT-2 Large (774M parameters)
    global_batch_size = 32
    micro_batch_size = 8
    sequence_length = 1024
    max_steps = 20
    learning_rate = 3e-4
    
    if rank == 0:
        print(f"\nFSDP Configuration:")
        print(f"Sharding strategy: {args.strategy}")
        print(f"Global batch size: {global_batch_size}")
        print(f"Micro batch size per rank: {micro_batch_size}")
        print(f"World size: {world_size}")
        print(f"Sequence length: {sequence_length}")
        print(f"Max steps: {max_steps}")
    
    # Deterministic seeding
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Model setup
    if rank == 0:
        print("Creating model...")
    
    model = MyGPT2LMHeadModel(config)
    model.init_weights()
    
    # Wrap with FSDP (only difference from DDP!)
    model = setup_fsdp_model(model, device, args.strategy)
    
    if rank == 0:
        print(f"Model wrapped with FSDP ({args.strategy})")
    
    # Data setup
    if rank == 0:
        print("Setting up distributed data...")
    
    files = download_fineweb10B_files(num_train_files=10)
    train_files = [f for f in files if "train" in str(f)]
    
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
        ddp_enabled=True  # FSDP also uses no_sync and distributed features
    )
    
    # Training with profiling
    if rank == 0:
        print(f"Starting FSDP training with {args.strategy}...")
    
    from torch.profiler import profile, ProfilerActivity, schedule
    
    # Save traces to strategy-specific directory
    trace_dir = f"outputs/traces/fsdp_{args.strategy.lower()}"
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = f"{trace_dir}/rank{rank}_trace.json"
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=lambda p: p.export_chrome_trace(trace_path)
    ) as prof:
        trainer.train(dataloader, profiler=prof)
    
    if rank == 0:
        print(f"FSDP training with {args.strategy} completed!")
        print(f"Traces saved to: {trace_dir}/")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
