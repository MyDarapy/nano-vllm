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
    

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.hidden_size = config.hidden_size # model dimension
        self.num_heads = config.num_attention_heads 
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads # how many query head share a KV head 

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads*self.head_dim, bias = False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads*self.head_dim, bias = False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads*self.head_dim, bias=False)

    def forward(
            self, 
            x, 
            ):
        batch_size, seq_len, embed_dim = x.shape

        q = self.proj(x)
        k = self.proj(x)
        v = self.proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # apply rope
        





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
