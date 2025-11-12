# PyTorch Distributed Training

## Initial Setup

This project uses `uv` for Python package management.

### Install uv

First, install `uv` by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Setup Project

```bash
# Install dependencies
uv sync

# Run Python scripts
uv run python main.py

# Run with environment variables
ENV_VARIABLE=value uv run python script.py
```

## Assignments

- [Assignment 0: Single GPU Training & Profiling](assignments/assignment0/README.md)
- [Assignment 1: Data Parallel Training (DDP/FSDP)](assignments/assignment1/README.md)
