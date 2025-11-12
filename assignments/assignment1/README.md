# Assignment 1: Data Parallel Training (DDP/FSDP)

## Setup

```bash
# Sync dependencies
uv sync
```

## Overview
Distributed training with DDP and FSDP, then analyze performance differences using HTA.

## Structure
```
model/              # GPT-2 model implementation
data/               # Data loaders (base + distributed)
train/              # Trainers (base + distributed)
train_baseline.py   # Single GPU training (reference)
train_ddp.py        # DDP training
train_fsdp.py       # FSDP training with different strategies
analyze_traces.ipynb # HTA trace analysis
traces/             # Generated profiler traces for analysis
```

## Tasks

### Part 1: Implement Distributed Components
Complete TODOs in:
- `data/distributed_data_loader.py` - Rank-based data partitioning
- `train/distributed_trainer.py` - Gradient synchronization, loss aggregation

### Part 2: DDP Training
Run DDP training and compare with baseline:
```bash
# Baseline
uv run python train_baseline.py

# DDP (2 GPUs)
uv run torchrun --nproc_per_node=2 train_ddp.py
```

### Part 3: FSDP Training
Train with different sharding strategies:
```bash
# FULL_SHARD - maximum memory savings
uv run torchrun --nproc_per_node=2 train_fsdp.py --strategy FULL_SHARD

# SHARD_GRAD_OP - only shard gradients/optimizer
uv run torchrun --nproc_per_node=2 train_fsdp.py --strategy SHARD_GRAD_OP
```

### Part 4: Analysis
Use `analyze_traces.ipynb` to compare:
- Temporal breakdown (compute vs communication vs idle)
- Communication patterns (all_reduce, all_gather, reduce_scatter)
- Memory usage

## Generated Outputs

All generated files are organized in dedicated directories:

**Training Traces** (`outputs/traces/`):
- `outputs/traces/baseline/` - Single GPU baseline traces
- `outputs/traces/ddp/` - DDP training traces (per-rank)
- `outputs/traces/fsdp_full_shard/` - FSDP FULL_SHARD traces (per-rank)
- `outputs/traces/fsdp_shard_grad_op/` - FSDP SHARD_GRAD_OP traces (per-rank)

**Training Data** (`.cache/data/`):
- `.cache/data/fineweb10B/` - Downloaded training dataset files
