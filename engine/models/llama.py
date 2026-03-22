import torch
import torch.nn as nn
import torch.nn.functional as F 


from engine.config import ModelConfig 
from engine.attention.gqa import GroupedQueryAttention
from engine.attention.flash_attention import TritonFlashAttention
from engine.attention.paged_attention import PagedAttention


class RoPE(nn.Module):
    """" Rotary Positional Embedding (RoPE)"""
    def __init__(self, ):
        super().__init__()
        self.

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight 


class LlamaMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.intermidate_size = config.intermidiate_size

        self.gate_proj = nn.Linear(self.hidden_dim, self.intermidate_size, bias=False)   
        self.up_proj = nn.Linear(self.hidden_dim, self.intermidate_size, bias=False)
        self.down_proj = nn.Linear(self.intermidate_size, self.hidden_dim, bias=False)


    def forward(self, x):
        # Swiglu = down(silu(gate(x) * up(x)))
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
    
    

class LlamaDecoder():
    def __init__(self, ):
        super().__init__()
        self.mlp = LlamaMLP()
