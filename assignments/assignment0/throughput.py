"""
Task 2: Throughput Measurement & Scaling Analysis

Measure training throughput and extrapolate for modern LLMs.
"""

import torch
import time
from model.my_gpt2 import MyGPT2LMHeadModel
from transformers import AutoConfig


def measure_tokens_per_second(model, batch_size, seq_length, num_steps=20, device='cuda'):
    """
    Measure actual training throughput.
    
    Steps:
    1. Create dummy training data
    2. Warmup runs (PyTorch/CUDA optimization)
    3. Timed training loop over num_steps
    4. Calculate tokens per second
    
    Args:
        model: GPT-2 model
        batch_size: Training batch size
        seq_length: Sequence length
        num_steps: Number of steps to measure over
        device: Device to run on
        
    Returns:
        float: Tokens per second achieved
    """
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Create dummy data
    input_ids = torch.randint(0, 50257, (batch_size, seq_length), device=device)
    targets = torch.randint(0, 50257, (batch_size, seq_length), device=device)
    
    # TODO: Tokens processed per batch
    tokens_per_batch = 0  # batch_size * seq_length
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        # Run training step: forward, backward, optimizer step
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Measure throughput
    print(f"Measuring throughput over {num_steps} steps...")
    
    torch.cuda.synchronize()  # Wait for all operations to complete
    start_time = time.time()
    
    for step in range(num_steps):
        # Training step (forward + backward + optimizer)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pass
    
    torch.cuda.synchronize()  # Wait for all GPU operations to finish
    end_time = time.time()
    
    # TODO: Calculate results
    elapsed_time = 0  # end_time - start_time
    total_tokens = 0  # num_steps * tokens_per_batch
    tokens_per_second = 0  # total_tokens / elapsed_time
    
    print(f"\n--- Throughput Results ---")
    print(f"Total tokens processed: {total_tokens:,}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.1f}")
    print(f"Steps per second: {num_steps/elapsed_time:.2f}")
    
    return tokens_per_second


def extrapolate_modern_training(tokens_per_sec_model, model):
    """
    As an exercise, let's try to extrapolate this setup to modern LLM training requirements.
    
    Key assumptions:
    - Same GPU type and efficiency
    - Linear scaling: 124M → 1T params ≈ 8000x slower tokens/sec  
    - Target: Train 1T model on 10T tokens
    - No parallelization (single GPU baseline)
    
    Args:
        tokens_per_sec_model: Measured tokens/sec for the model
        model: Model used
    """
    
    # Model scaling parameters
    gpt2_small_params = sum(p.numel() for p in model.parameters())
    modern_llm_params = 1e12       # 1 trillion parameters  
    training_tokens = 10e12        # 10 trillion tokens
    
    # TODO: Calculate scaling factor
    # Assumption: FLOPs scale roughly linearly with parameter count
    # So tokens/sec scales inversely with parameter count
    scaling_factor = 0  # modern_llm_params / gpt2_small_params
    
    # TODO: Estimate modern LLM throughput
    tokens_per_sec_1t = 0  # tokens_per_sec_124m / scaling_factor
    
    # TODO: Calculate training time
    training_time_seconds = 0  # training_tokens / tokens_per_sec_1t
    training_time_days = training_time_seconds / (24 * 3600)
    training_time_years = training_time_days / 365
    
    print(f"\n--- Modern LLM Training Extrapolation ---")
    print(f"GPT-2 Small: {tokens_per_sec_model:,.0f} tokens/sec")
    print(f"1T Model throughput: {tokens_per_sec_1t:.3f} tokens/sec")
    print(f"")
    print(f"Training 10T tokens:")
    print(f"  Time: {training_time_days:,.0f} days ({training_time_years:.1f} years)")
    
    return {
        'tokens_per_sec_1t': tokens_per_sec_1t,
        'training_days': training_time_days,
    }


def compare_batch_sizes():
    """
    BONUS - Compare throughput across different batch sizes.
    
    Measure how batch size affects tokens/sec to understand GPU utilization.
    """
    
    config = AutoConfig.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_length = 1024
    
    batch_sizes = [1, 4, 8, 16, 32, 64]  # Adjust based on your GPU memory
    results = []
    
    print(f"\n--- Batch Size Analysis ---")
    
    for batch_size in batch_sizes:
        try:
            print(f"\nTesting batch_size={batch_size}...")
            
            # Create model for each test (fresh start)
            model = MyGPT2LMHeadModel(config).to(device)
            
            # TODO: Measure throughput
            tokens_per_sec = measure_tokens_per_second(model, batch_size, seq_length, num_steps=20, device=device)
            
            results.append({
                'batch_size': batch_size,
                'tokens_per_sec': tokens_per_sec,
                'memory_mb': torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            })
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM at batch_size={batch_size}")
                break
            else:
                raise e
    
    print(f"\n--- Batch Size Results ---")
    print(f"{'Batch Size':<12} {'Tokens/sec':<12} {'Memory (MB)':<12}")
    print("-" * 40)
    for result in results:
        print(f"{result['batch_size']:<12} {result['tokens_per_sec']:<12.0f} {result['memory_mb']:<12.0f}")
    
    return results


def main():
    """Run the throughput measurement task."""
    
    print("Task 2: Throughput Measurement & Scaling")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU. Throughput will be much slower.")
    
    # Configuration
    config = AutoConfig.from_pretrained("gpt2")
    batch_size = 8
    seq_length = 1024
    
    # TODO: Create model
    # model = MyGPT2LMHeadModel(config).to(device)
    model = None
    
    print(f"Model: GPT-2 Small ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    
    # Measure throughput
    tokens_per_sec = measure_tokens_per_second(model, batch_size, seq_length, device=device)
    
    # Extrapolate to modern LLMs
    extrapolate_modern_training(tokens_per_sec)
    
    # Bonus: Compare batch sizes
    compare_batch_sizes()


if __name__ == "__main__":
    main()
