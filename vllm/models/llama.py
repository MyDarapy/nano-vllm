import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from typing import Tuple, Union

from vllm.config import ModelConfig 
from vllm.core.cache import BlockKCache, KVCache

try:
    from vllm.kernels.paged_prefill import TritonFlashAttention
except Exception:
    TritonFlashAttention = None

try:
    from vllm.kernels.paged_decode import PagedFlashAttention
except Exception:
    PagedFlashAttention = None

try:
    from vllm.kernels.mlp import MLP
except Exception:
    MLP = None

try:
    from vllm.core.kv_scatter import store_kvcache
except Exception:
    store_kvcache = None


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
        self.flash_attention = TritonFlashAttention() if TritonFlashAttention is not None else None
        self.paged_attention = PagedFlashAttention() if PagedFlashAttention is not None else None

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        return x.repeat_interleave(self.num_kv_groups, dim=1)

    def _reference_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal:
            q_len = q.shape[-2]
            k_len = k.shape[-2]
            causal_mask = torch.tril(
                torch.ones((q_len, k_len), device=q.device, dtype=torch.bool)
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(probs, v)

    def _get_position_ids(
        self,
        metadata,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if metadata.positions is None:
            return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        if metadata.positions.dim() == 1:
            return metadata.positions.unsqueeze(0)
        return metadata.positions

    def forward(self, x, metadata, kv_cache: Union[KVCache, BlockKCache]):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x) # [B, S, H_q * head_dim]
        k = self.k_proj(x) # [B, S, H_kv * head_dim]
        v = self.v_proj(x) # [B, S, H_kv * head_dim]
  
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        position_ids = self._get_position_ids(metadata, batch_size, seq_len, x.device)
        cos, sin = self.rotary_emb(q, position_ids)

        q = q.transpose(1, 2).contiguous()  # [B, Hq, S, D]
        k = k.transpose(1, 2).contiguous()  # [B, Hkv, S, D]
        v = v.transpose(1, 2).contiguous()  # [B, Hkv, S, D]

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if isinstance(kv_cache, BlockKCache):
            if store_kvcache is None or self.paged_attention is None:
                raise RuntimeError(
                    "Paged attention dependencies are unavailable. Install triton to use BlockKCache mode."
                )
            k_for_cache = k.transpose(1, 2).contiguous().view(-1, self.num_kv_heads, self.head_dim)
            v_for_cache = v.transpose(1, 2).contiguous().view(-1, self.num_kv_heads, self.head_dim)

            if metadata.slot_mapping is None:
                raise ValueError("slot_mapping is required when using BlockKCache")

            store_kvcache(
                self.layer_idx,
                k_for_cache,
                v_for_cache,
                kv_cache,
                metadata.slot_mapping.reshape(-1),
            )

            if metadata.is_prefill:
                if self.flash_attention is None:
                    raise RuntimeError(
                        "Flash attention dependencies are unavailable. Install triton to use paged prefill."
                    )
                attn_output = self.flash_attention.flash_attention(q, k, v, metadata.context_lens, causal=True)
            else:
                q_step = q[:, :, -1, :]
                attn_output = self.paged_attention.paged_decode_attn(
                    q_step,
                    kv_cache,
                    self.layer_idx,
                    metadata.block_tables,
                    metadata.context_lens,
                ).unsqueeze(2)
        else:
            keys, values = kv_cache.update(self.layer_idx, k, v)
            attn_output = self._reference_attention(
                q,
                keys,
                values,
                causal=metadata.is_prefill,
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        return self.o_proj(attn_output.view(batch_size, seq_len, -1))


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.intermidate_size = config.intermidiate_size

        self.fused_mlp = MLP() if MLP is not None else None

        self.gate_proj = nn.Linear(self.hidden_dim, self.intermidate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermidate_size, bias=False)
        self.down_proj = nn.Linear(self.intermidate_size, self.hidden_dim, bias=False)

    def forward(self, x):
        if self.fused_mlp is not None:
            w_gate = self.gate_proj.weight.t().contiguous()
            w_up = self.up_proj.weight.t().contiguous()
            w_down = self.down_proj.weight.t().contiguous()

            hidden = self.fused_mlp.ffn_stage_1(x, w_gate=w_gate, w_up=w_up)
            output = self.down_proj(hidden)
            #output = self.fused_mlp.ffn_stage_2(hidden, w_down=w_down)
            return output

        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class LlamaMLPVanilla(nn.Module):
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
        self.mlp = LlamaMLPVanilla(config)
        self.attention = LlamaAttention(config, layer_idx)
        self.pre_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, metadata, kv_cache: Union[KVCache, BlockKCache]):
        residual = x 
        x = self.pre_layernorm(x)
        x = residual + self.attention(x, metadata, kv_cache)
        residual = x 
        x = self.post_layernorm(x)
        x = residual + self.mlp(x)
        return x 
    

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(LlamaDecoderLayer(config, layer_idx)
                                    for layer_idx in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, metadata, kv_cache: Union[KVCache, BlockKCache]):
        x = self.embed_tokens(input_ids)
        if isinstance(kv_cache, KVCache):
            kv_cache.begin_forward(input_ids.shape[1])
        try:
            for layer in self.layers:
                x = layer(x, metadata, kv_cache)
        finally:
            if isinstance(kv_cache, KVCache):
                kv_cache.end_forward()

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits 