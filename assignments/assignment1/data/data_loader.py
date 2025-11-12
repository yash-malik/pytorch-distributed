import glob
import torch

from pathlib import Path
from typing import Iterator, Tuple, List, Union
from huggingface_hub import hf_hub_download


def download_fineweb10B_files(local_dir=".cache/data/fineweb10B", num_train_files=None):
    """
    Download fineweb10B-gpt2 dataset files from HuggingFace Hub
    https://huggingface.co/datasets/kjj0/fineweb10B-gpt2 (contains the GPT-2 tokens for fineweb10B)
    
    Args:
        local_dir: Local directory to save files
        num_train_files: Number of training files to download (1-103), None for all
    
    Returns:
        List of downloaded file paths
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    
    # Download validation file
    print("Downloading validation file...")
    val_file = "fineweb_val_000000.bin"
    val_path = local_dir / val_file
    
    if not val_path.exists():
        print(f"  Downloading {val_file}...")
        hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2",
            filename=val_file,
            repo_type="dataset",
            local_dir=local_dir
        )
    else:
        print(f"  {val_file} already exists")
    downloaded_files.append(val_path)
    
    # Download training files
    if num_train_files is None:
        num_train_files = 103  # Full dataset has 103 training files
    
    print(f"Downloading {num_train_files} training files...")
    for i in range(1, num_train_files + 1):
        train_file = f"fineweb_train_{i:06d}.bin"
        train_path = local_dir / train_file
        
        if not train_path.exists():
            print(f"  Downloading {train_file}... ({i}/{num_train_files})")
            hf_hub_download(
                repo_id="kjj0/fineweb10B-gpt2",
                filename=train_file,
                repo_type="dataset",
                local_dir=local_dir
            )
        else:
            print(f"  {train_file} already exists ({i}/{num_train_files})")
        downloaded_files.append(train_path)
    
    print(f"Downloaded {len(downloaded_files)} files to {local_dir}")
    return downloaded_files


class KJJ0DataLoader:
    """
    Format of kjj0 .bin files:
        - Header: 256 int32 values (1024 bytes)
        - header[0] = 20240520 (magic number)
        - header[1] = 1 (version) 
        - header[2] = number of tokens
        - Tokens: sequence of uint16 values
    
    Reads .bin files sequentially and yields batches of token sequences.
    Each sequence consists of (input_tokens, target_tokens) where targets 
    are inputs shifted by 1 position for next-token prediction.
    
    Args:
        file_paths: List of .bin file paths (e.g., from download_fineweb10B_files() output)
        batch_size: Number of sequences per batch
        sequence_length: Length of each sequence
        device: Device to place tensors on ('cuda' or 'cpu')
    """
    
    def __init__(self, file_paths: List[Union[str, Path]], batch_size: int, sequence_length: int, device: str = 'cpu'):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device
        
        # Convert Path objects to strings and sort
        self.files = sorted([str(f) for f in file_paths])
        assert self.files, "Empty file list provided"
        
        print(f"Found {len(self.files)} shard files")
        
        # Initialize state
        self.current_shard_idx = 0
        self.current_tokens = None
        self.current_position = 0
        
    def _load_shard(self, filepath: str) -> torch.Tensor:
        """
        Load tokens from a single .bin file.
        
        Returns:
            torch.Tensor of tokens as uint16
        """
        print(f"Loading shard: {filepath}")
        
        with open(filepath, 'rb') as f:
            # Read header (256 int32 values = 1024 bytes)
            header_bytes = f.read(256 * 4)
            header = torch.frombuffer(header_bytes, dtype=torch.int32)
            
            # Validate header
            magic_number = header[0]
            version = header[1] 
            token_count = header[2]
            
            assert magic_number == 20240520, f"Invalid magic number: {magic_number}, expected 20240520"
            assert version == 1, f"Unsupported version: {version}, expected 1"
    
            print(f"  Tokens in shard: {token_count:,}")
            
            # Read tokens (uint16)
            token_bytes = f.read(token_count * 2)
            tokens = torch.frombuffer(token_bytes, dtype=torch.uint16)
            
            if len(tokens) != token_count:
                raise ValueError(f"Token count mismatch: got {len(tokens)}, expected {token_count}")
                
            return tokens
    
    def _get_next_sequence(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the next sequence of tokens.
        
        Returns:
            (input_tokens, target_tokens) where targets are inputs shifted by 1
        """
        # Load next shard if needed
        while self.current_tokens is None or self.current_position + self.sequence_length >= len(self.current_tokens):
            if self.current_shard_idx >= len(self.files):
                raise StopIteration("No more data available")
                
            self.current_tokens = self._load_shard(self.files[self.current_shard_idx])
            self.current_shard_idx += 1
            self.current_position = 0
        
        # Extract sequence
        start_pos = self.current_position
        end_pos = start_pos + self.sequence_length + 1  # +1 for target
            
        sequence = self.current_tokens[start_pos:end_pos]
        self.current_position += self.sequence_length
        
        # Split into input and target
        inputs = sequence[:-1]
        targets = sequence[1:]
        
        return inputs, targets
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over batches of sequences.
        
        Yields:
            (input_batch, target_batch) tensors of shape [batch_size, sequence_length]
        """
        # Reset state for new iteration
        self.current_shard_idx = 0
        self.current_tokens = None
        self.current_position = 0
        
        while True:
            try:
                # Collect batch_size sequences
                input_batch = []
                target_batch = []
                
                for _ in range(self.batch_size):
                    inputs, targets = self._get_next_sequence()
                    input_batch.append(inputs)
                    target_batch.append(targets)
                
                input_tensor = torch.stack(input_batch).long().to(self.device)
                target_tensor = torch.stack(target_batch).long().to(self.device)
                
                yield input_tensor, target_tensor
                
            except StopIteration:
                break
    
    def get_total_tokens(self) -> int:
        """
        Get the total number of tokens across all shards.
        """
        total = 0
        for filepath in self.files:
            with open(filepath, 'rb') as f:
                header_bytes = f.read(256 * 4)
                header = torch.frombuffer(header_bytes, dtype=torch.int32)
                total += header[2]  # token_count
        return total
    
    def get_info(self) -> dict:
        """
        Get information about the dataset.
        """
        return {
            'num_shards': len(self.files),
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length,
            'device': self.device,
            'files': self.files,
            'total_tokens': self.get_total_tokens(),
        }
