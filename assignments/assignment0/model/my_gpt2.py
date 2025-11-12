import torch
from torch import nn
import functools
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts
from transformers.activations import ACT2FN
from transformers import AutoConfig, AutoModelForCausalLM
from pytorch_utils import compute_intensive_ops


class MyGPT2Attention(nn.Module):
    """Multi-head self-attention mechanism for GPT-2."""
    
    def __init__(self, config):
        super().__init__()
        
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        
        # Merged Q, K, V weight matrices
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        # Output projection layer
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        # Attention and residual dropout
        self.attn_dropout = nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)
        
        # Causal mask buffer
        max_positions = config.n_ctx
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_sz, seq_len = hidden_states.shape[:2]
        
        # Compute Q, K, V
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        
        # Reshape for multi-head attention: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = q.reshape(batch_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention output
        attn_output = self._compute_attention(q, k, v)
        attn_output = attn_output.reshape(batch_sz, seq_len, -1)
        
        # Apply output projection and residual dropout
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output
    
    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute scaled dot-product attention with causal masking."""
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask
        seq_len = q.size(-2)
        attn_weights = attn_weights.masked_fill_(
            ~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf")
        )
        
        # Apply softmax and dropout
        attn_scores = torch.softmax(attn_weights, dim=-1)
        attn_scores = self.attn_dropout(attn_scores)
        
        # Apply attention to values and transpose back
        attn_output = torch.matmul(attn_scores, v)
        return attn_output.transpose(1, 2)  # (B, T, num_heads, head_dim)


class MyGPT2MLP(nn.Module):
    """Feed-forward network for GPT-2."""
    
    def __init__(self, config):
        super().__init__()
        
        self.embed_dim = config.n_embd
        
        # Up projection, activation, down projection
        self.c_fc = nn.Linear(self.embed_dim, 4 * self.embed_dim, bias=True)
        self.act = ACT2FN[config.activation_function]
        self.c_proj = nn.Linear(4 * self.embed_dim, self.embed_dim, bias=True)
        self.dropout = nn.Dropout(p=config.resid_pdrop)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MyGPT2Block(nn.Module):
    """Transformer block with pre-norm architecture."""
    
    def __init__(self, config):
        super().__init__()
        
        self.embed_dim = config.n_embd
        
        self.ln_1 = nn.LayerNorm(
            normalized_shape=self.embed_dim,
            eps=config.layer_norm_epsilon
        )
        self.attn = MyGPT2Attention(config)
        self.ln_2 = nn.LayerNorm(
            normalized_shape=self.embed_dim,
            eps=config.layer_norm_epsilon
        )
        self.mlp = MyGPT2MLP(config)
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Attention block with residual connection
        residual = hidden_state
        hidden_state = self.ln_1(hidden_state)
        attn_output = self.attn(hidden_state)
        hidden_state = attn_output + residual
        
        # MLP block with residual connection
        residual = hidden_state
        hidden_state = self.ln_2(hidden_state)
        mlp_output = self.mlp(hidden_state)
        hidden_state = mlp_output + residual
        
        return hidden_state


class MyGPT2Model(nn.Module):
    """GPT-2 transformer model."""
    
    def __init__(self, config, enable_activation_checkpoint=True):
        super().__init__()
        
        self.enable_activation_checkpoint = enable_activation_checkpoint
        # Create context function for selective checkpointing
        self.context_fn = functools.partial(create_selective_checkpoint_contexts, compute_intensive_ops)
    
        # Token and position embeddings
        self.wte = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd)
        self.wpe = nn.Embedding(num_embeddings=config.n_ctx, embedding_dim=config.n_embd)
        
        # Embedding dropout
        self.drop = nn.Dropout(p=config.embd_pdrop)
        
        # Transformer blocks
        self.h = nn.ModuleList([MyGPT2Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(
            normalized_shape=config.n_embd,
            eps=config.layer_norm_epsilon
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        batch_size, seq_len = input_ids.shape
        input_embed = self.wte(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        pos_embed = self.wpe(pos_ids)
        
        # Combine embeddings and apply dropout
        hidden_state = input_embed + pos_embed
        hidden_state = self.drop(hidden_state)
        
        # Pass through transformer blocks with optional selective checkpointing
        for block in self.h:
            if self.enable_activation_checkpoint and self.training:
                hidden_state = checkpoint(
                    block, hidden_state, 
                    use_reentrant=False,
                    context_fn=self.context_fn
                )
            else:
                hidden_state = block(hidden_state)
        
        # Apply final layer norm
        hidden_state = self.ln_f(hidden_state)
        
        return hidden_state


class MyGPT2LMHeadModel(nn.Module):
    """GPT-2 model with language modeling head."""
    
    def __init__(self, config, enable_activation_checkpoint=True):
        super().__init__()
        
        self.transformer = MyGPT2Model(config, enable_activation_checkpoint=enable_activation_checkpoint)
        
        # Language modeling head with tied weights
        self.lm_head = nn.Linear(
            in_features=config.n_embd, 
            out_features=config.vocab_size, 
            bias=False
        )
        # Tie weights between embedding and LM head
        self.lm_head.weight = self.transformer.wte.weight
        
        # Initialize weights using the GPT-2 initialization strategy
        self.init_weights()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_state = self.transformer(input_ids)
        logits = self.lm_head(hidden_state)
        return logits
    
    def init_weights(self):
        """Initialize model weights following OpenAI GPT-2 initialization scheme.
        
        Based on: https://github.com/openai/gpt-2/blob/master/src/model.py
        - All linear layer weights: normal(mean=0, std=0.02)
        - All biases: zeros
        - Token embeddings (wte): normal(mean=0, std=0.02)  
        - Position embeddings (wpe): normal(mean=0, std=0.01)
        - LayerNorm weights: ones, LayerNorm biases: zeros
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # All linear layers use std=0.02 (no special scaling for residual projections)
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Embedding):
                if module == self.transformer.wte:
                    # Token embeddings: std=0.02
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif module == self.transformer.wpe:
                    # Position embeddings: std=0.01  
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm: weights=1, biases=0
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def save(self, path: str) -> None:
        """Save model state dict."""
        torch.save(self.state_dict(), path)
    
    @staticmethod
    def _convert_conv1d_to_linear_state_dict(state_dict):
        """Convert Conv1D weights to Linear weights format.
        
        Conv1D weight shape: (nx, nf) where nx=in_features, nf=out_features
        Linear weight shape: (out_features, in_features)
        
        The key difference is that Conv1D weights need to be transposed for Linear layers.
        """
        converted_state_dict = {}
        
        for key, value in state_dict.items():
            if any(layer_name in key for layer_name in ['c_attn', 'c_proj', 'c_fc']):
                if 'weight' in key:
                    # Conv1D weights are stored as (nx, nf) but Linear expects (out_features, in_features)
                    # Conv1D performs: output = input @ weight + bias
                    # Linear performs: output = input @ weight.T + bias
                    # So we need to transpose the Conv1D weight to match Linear's expected format
                    converted_state_dict[key] = value.t().contiguous()
                else:
                    # Bias remains the same
                    converted_state_dict[key] = value
            else:
                # All other parameters remain unchanged
                converted_state_dict[key] = value
                
        return converted_state_dict

    @staticmethod
    def from_pretrained(state_dict, config=None, enable_activation_checkpoint=True) -> 'MyGPT2LMHeadModel':
        """Load model from our model's state dict."""
        if config is None:
            config = AutoConfig.from_pretrained("gpt2")
        
        model = MyGPT2LMHeadModel(config, enable_activation_checkpoint=enable_activation_checkpoint)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def from_hf_pretrained(model_name="gpt2", enable_activation_checkpoint=True) -> 'MyGPT2LMHeadModel':
        """Load model from HuggingFace pretrained weights with Conv1D conversion."""
        print(f"Downloading HuggingFace {model_name} model...")
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        hf_state_dict = hf_model.state_dict()
        
        # Convert Conv1D weights to Linear format
        converted_state_dict = MyGPT2LMHeadModel._convert_conv1d_to_linear_state_dict(hf_state_dict)
        
        config = AutoConfig.from_pretrained(model_name)
        
        model = MyGPT2LMHeadModel(config, enable_activation_checkpoint=enable_activation_checkpoint)
        model.load_state_dict(converted_state_dict)
        return model
    
    @classmethod
    def from_hf_config(cls, model_name="gpt2", enable_activation_checkpoint=True) -> 'MyGPT2LMHeadModel':
        """Create model with HuggingFace GPT-2 config."""
        config = AutoConfig.from_pretrained(model_name)
        return cls(config, enable_activation_checkpoint=enable_activation_checkpoint)
