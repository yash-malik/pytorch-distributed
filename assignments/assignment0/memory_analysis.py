"""
Task 1: Memory Analysis
"""

import torch
from pathlib import Path

from model.my_gpt2 import MyGPT2LMHeadModel
from transformers import AutoConfig


def calculate_memory_breakdown(config, batch_size, seq_length):
    """
    Calculate rough memory requirements.
    
    Memory Components (assuming fp32, 4 bytes per parameter):
    1. Model Parameters: total_params * 4 bytes 
    2. Gradients: same size as parameters  
    3. Optimizer States: 2 * param_size (Adam momentum + variance)
    
    Note: Activation memory is ignored because activation checkpointing is enabled.
    
    Args:
        config: GPT-2 model configuration
        batch_size: Training batch size
        seq_length: Sequence length
        
    Returns:
        dict: Memory breakdown in MB
    """
    
    # TODO: Create model to get parameter count
    # Hint: model = MyGPT2LMHeadModel(config)
    #       total_params = sum(p.numel() for p in model.parameters())
    model = None
    total_params = 0
    
    print(f"Model parameters: {total_params:,}")
    
    # TODO: Calculate memory components
    param_memory_mb = 0      # total_params * 4 / (1024**2)
    gradient_memory_mb = 0   # same as param_memory_mb  
    optimizer_memory_mb = 0  # 2 * param_memory_mb (Adam momentum + variance)
    
    breakdown = {
        'parameters_mb': param_memory_mb,
        'gradients_mb': gradient_memory_mb,
        'optimizer_states_mb': optimizer_memory_mb,
    }
    breakdown['total_mb'] = sum(breakdown.values())
    
    return breakdown


def profile_actual_memory(config, batch_size, seq_length):
    """
    Profile actual memory usage during training.
    
    Steps:
    1. Start memory recording
    2. Create model, optimizer, dummy data
    3. Run a few training steps
    4. Measure peak memory
    5. Save snapshot for visualization
    
    Returns:
        dict: Actual memory statistics
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TODO: Start memory profiling
    torch.cuda.memory._record_memory_history(enabled='all')
    
    try:
        # TODO: Create model and move to device
        # model = MyGPT2LMHeadModel(config).to(device)
        model = None
        
        # TODO: Create optimizer
        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer = None
        
        # TODO: Create dummy training data
        # input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
        # targets = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
        input_ids = None
        targets = None
        
        # TODO: Run training steps to allocate gradient/optimizer memory
        for step in range(3):
            # Forward pass
            # logits = model(input_ids)
            # loss = torch.nn.functional.cross_entropy(
            #     logits.view(-1, logits.size(-1)), 
            #     targets.view(-1)
            # )
            
            # Backward pass
            # loss.backward()
            # optimizer.step() 
            # optimizer.zero_grad()
            pass
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
            reserved_memory = torch.cuda.memory_reserved() / (1024**2)  # MB
        else:
            peak_memory = 0
            reserved_memory = 0
        
        # TODO: Save memory snapshot for visualization
        torch.cuda.memory._dump_snapshot("task1_memory_snapshot.pickle")
        print("Memory snapshot saved: task1_memory_snapshot.pickle")
        print("View at: https://pytorch.org/memory_viz")
        
        return {
            'peak_memory_mb': peak_memory,
            'reserved_memory_mb': reserved_memory
        }
        
    finally:
        # Stop memory recording
        torch.cuda.memory._record_memory_history(enabled=None)


def main():
    """Run the memory analysis task."""
    
    print("Task 1: Memory Analysis")
    print("="*50)
    
    # Configuration
    config = AutoConfig.from_pretrained("gpt2")
    batch_size = 8
    seq_length = 1024
    
    print(f"Model: GPT-2 Small")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Estimate rough memory requirements
    estimated = calculate_memory_breakdown(config, batch_size, seq_length)
    
    # Profile actual memory usage
    actual = profile_actual_memory(config, batch_size, seq_length)
    
    # Analyze results
    print(f"\n--- Memory Breakdown ---")
    print(f"Estimated Memory:")
    print(f"  Parameters:       {estimated['parameters_mb']:.2f} MB")
    print(f"  Gradients:        {estimated['gradients_mb']:.2f} MB")
    print(f"  Optimizer States: {estimated['optimizer_states_mb']:.2f} MB")
    print(f"  Total Estimated:  {estimated['total_mb']:.2f} MB")

    print(f"\nActual Memory:")
    print(f"  Peak Allocated:   {actual['peak_memory_mb']:.2f} MB")
    print(f"  Reserved:         {actual['reserved_memory_mb']:.2f} MB")
    print(f"\nDifference:")
    print(f"  Allocated vs Estimated: {actual['peak_memory_mb'] - estimated['total_mb']:.2f} MB")
    

if __name__ == "__main__":
    main()
