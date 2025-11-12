# Assignment 0: Single GPU Training & Profiling

## Setup

```bash
# Sync dependencies
uv sync
```

## Overview
Profile GPU memory usage and measure training throughput for GPT2 models.

## Objectives
- Calculate and validate memory estimates with PyTorch profiler
- Measure training throughput (tokens/sec)

## Tasks

### Task 1: Memory Analysis
Calculate and profile GPU memory usage during training.

```bash
uv run python memory_analysis.py
```

**Key Concepts:**
- Memory components: parameters, gradients, optimizer states (Adam)
- Validation with PyTorch memory snapshot (view at pytorch.org/memory_viz)

### Task 2: Throughput & Scaling
Measure training throughput and extrapolate to modern LLM requirements.

```bash
uv run python throughput.py
```

**Key Concepts:**
- Tokens per second measurement
- Scaling analysis: 124M -> 1T parameters
- Understanding computational requirements for modern LLMs

## Files
- `memory_analysis.py` - Memory profiling assignment
- `throughput.py` - Throughput measurement and scaling
- `model/my_gpt2.py` - Self-contained GPT-2 implementation

## Generated Outputs

All generated files are saved to the `outputs/` directory:

- `outputs/task1_memory_snapshot.pickle` - Memory snapshot from Task 1 (view at [pytorch.org/memory_viz](https://pytorch.org/memory_viz))

