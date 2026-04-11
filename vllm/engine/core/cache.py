""""Exposes two different kinds of KV cache implementations for AR generation

- preallocated KV cache 
- BlockKV cache based on pagedattention

"""


from vllm.engine.core.block import BLOCK_SIZE
from vllm.engine.config import ModelConfig

import time
import torch


class KVCache:
    """Pre allocated kv cache for a single sequence. 
    (wastefu) causes internal and external fragmentation """
    def __init__(
            self, 
            config, 
            max_seq_len=512, 
            device="cuda", 
            dtype=torch.float16):
        
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.seq_len = 0 # represents how many tokens are already stored in the cache 

        self._write_positions = 0
        self._pending_seq_len = 0 

        # pre-allocate the cache
        self.cache = torch.zeros(self.num_layers, 2, self.num_kv_heads, self.max_seq_len, self.head_dim)


    def begin_forward(self, new_seq_len):

        """This saves the current write position so all layers write to the same token positions"""

        self._write_positions = self.seq_len # start writing new tokens after existing ones
        self._pending_seq_len = new_seq_len

        """A given KV state  needs to write to the same token position across different layers
        We freeze the write position for the entire forward pass of a sequence so all layers use the same write position
        Each layer gets its own KV but positions doesn't change
        Cache shape: [num_layers, 2, num_kv_heads, max_seq_len, head_dim] """
        
    def end_forward(self):
        # after a forward pass the KV cache sequence length is updated
         self.seq_len = self._write_positions + self._pending_seq_len
        
    def update(self, layer_idx, key_states, value_states):
        new_seq_len = key_states.shape[2]
        write_pos = self._write_positions

        self.cache[layer_idx, 0, :, write_pos : write_pos + new_seq_len, :] = key_states[0] # [1, num_kv_heads, seq_len, head_dim]
        self.cache[layer_idx, 1, :, write_pos: write_pos + new_seq_len, :] = value_states[0]

        # Each sequence owns the KV cache. No sharing. 
        end_pos = write_pos + new_seq_len
        keys = self.cache[layer_idx, 0, :, :end_pos, :].unsqueeze(0)
        values = self.cache[layer_idx, 1, :, :end_pos, :].unsequeeze(0)

        return keys, values
    
    def reset(self):
        self.seq_len=0
        self._write_positions = 0 
        self._pending_seq_len = 0 

    @property
    def memory_usage_mb(self):
        "Return sequence length memory usage in MB"
        return self.cache.element_size() * self.cache.numel() / (1024 * 1024)

def gather_kv_cache(caches, layer_idx):
    """Gather the KV states from multiple sequence cache for batched attention"""
    if not caches:
        raise ValueError("No cache provided")
    
    batch_size = len(caches)
    max_len = max(cache.seq_len for cache in caches)

    first = caches[0]
    num_kv_heads = first.num_kv_heads
    head_dim = first.head_dim
    device = first.device
    dtype = first.dtype


    keys = torch.zeros((batch_size, num_kv_heads, max_len, head_dim), 
                      device=device,
                      dtype=dtype)
    

    values = torch.zeros((batch_size, num_kv_heads, max_len, head_dim), 
                      device=device,
                      dtype=dtype)
    
    for i, cache in enumerate(caches):
        seq_len = cache.seq_len
        keys[i, :, :seq_len, :] = cache.cache[layer_idx, 0, :, :seq_len, :]
        values[i, :, :seq_len, :] = cache.cache[layer_idx, 1, :, :seq_len, :]

class BlockKCache():
    """A single pool of block. Sequences references this via their block tablee"""
    def _init__(self, num_blocks, 
                num_layers, block_size, 
                num_kv_heads, head_dim, 
                device, dtype):
        
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype


        self.key_cache = torch.zeros(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype)
        
        self.value_cache = torch.zeros(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype)
        
        def get_layer_caches(self, layer_id):
            "Get K nd V cache for a particular layer"
            return self.key_cache[layer_id], self.value_cache[layer_id]
        
        @property
        def memory_usage_mb(self):
            key_bytes = self.key_cache.element_size()*self.key_cache.numel()
            value_bytes = self.value_cache.element_size() * self.value_cache.numel()
            return (key_bytes + value_bytes) / (1024, 1024)
        

        @classmethod
        def from_config(
            cls, config,
            num_blocks, 
            block_size=BLOCK_SIZE,
            device = "cuda",
            dtype = torch.float16
        ):
            
            "create/initalize a BlockKVCache from model config"

            return cls(num_blocks=num_blocks,
                       num_layers=config.num_hidden_layers,
                       block_size=block_size,
                       num_kv_heads=config.num_kv_heads,
                       head_dim = config.head_dim,
                       device=device,
                       dtype = dtype
            )
        
        def __repr__(self):
            return(
                f"BlockKVCache(blocks={self.num_blocks}, "
                f"layers={self.num_layers},"
                f"block_size={self.block_size}, "
                f"memory={self.memeory_usage_mb: .1f}MB)"
            )