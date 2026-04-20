import torch
import torch.nn as nn
import torch.nn.functional as F 

import math 

from vllm.config import ModelConfig 
from vllm.attention.flash_attention import TritonFlashAttention
from vllm.attention.paged_attention import paged_decode_attn
from vllm.core.kv_scatter import store_kvcache
from vllm.core.cache import BlockKCache


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight 

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies: theta_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos and sin for all positions
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        """Pre-compute cos and sin values for positions 0 to seq_len-1."""
        positions = torch.arange(seq_len).float()
        # Outer product: [seq_len] x [dim/2] -> [seq_len, dim/2]
        freqs = torch.outer(positions, self.inv_freq)
        # Duplicate for pairs: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given positions."""
        # x: [batch, seq_len, num_heads, head_dim]
        # position_ids: [batch, seq_len]
        cos = self.cos_cached[position_ids]  # [batch, seq_len, dim]
        sin = self.sin_cached[position_ids]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input.

    For input [x1, x2, x3, x4], returns [-x3, -x4, x1, x2].
    This is used in the RoPE rotation formula.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    The rotation formula is:
    q_rotated = q * cos + rotate_half(q) * sin
    """
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [batch, seq_len, head_dim]
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config 
        self.hidden_size = config.hidden_size # model dimension
        self.num_heads = config.num_attention_heads 
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads # how many query head share a KV head (num_queries_per_kv)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads*self.head_dim, bias = False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads*self.head_dim, bias = False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads*self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads*self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings,
                                          base=config.rope_theta,)
        
        self.layer_idx = layer_idx
        self.flash_attention= TritonFlashAttention()
        #self.paged_attention = FlashPaged()


    def forward(self, x, metadata, kv_cache: BlockKCache):
        batch_size, seq_len, _ = x.shape

        q = self.proj(x) # [B, S, H_q * head_dim]
        k = self.proj(x) # [B, S, H_kv * head_dim]
        v = self.proj(x) # [B, S, H_kv * head_dim]
  

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q, k)
        k_for_cache = k.view(-1, self.num_kv_heads, self.head_dim)
        v_for_cache = v.view(-1, self.num_kv_heads, self.head_dim)
        
        assert metadata.slot_mapping is not None, "slot mapping needed for KV scatter"
        
        store_kvcache(self.layer_idx, 
                      k_for_cache,
                      v_for_cache, 
                      kv_cache,
                      metadata.slot_mapping)
        
        if metadata.is_prefill:
            #q = q.permute(0, 2, 1, 3).contiguous()
            #k = k.permute(0, 2, 1, 3).contiguous()
            #v = v.permute(0, 2, 1, 3).contiguous()

            attn_output = self.flash_attention(q, k, v, causal=True)
        else:
            q_step = q.squeeze(1)
            attn_output = self.paged_attention(q_step, kv_cache, metadata.block_tables,
                                               metadata.context_lens)
            attn_output = attn_output.unsqueeze(1)
        
        return self.o_proj(attn_output.view(batch_size, seq_len, -1))

        


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
    
    

class LlamaDecoderLayer(nn.Module):
    """Single transformer decoder layer"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.mlp = LlamaMLP(config)
        self.attention = LlamaAttention(config, layer_idx)
        self.pre_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, metadata, kv_cache: BlockKCache):
        residual = x 
        x = self.pre_layernorm(x)
        x = residual + self.attention(x, metadata, kv_cache)
        residual = x 
        x = self.post_layernorm(x)
        x = residual + self.mlp(x)
        return x 
    

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx)] 
                                    for layer_idx in config.num_hidden_layers)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, metadata, kv_cache):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, metadata, kv_cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits 

