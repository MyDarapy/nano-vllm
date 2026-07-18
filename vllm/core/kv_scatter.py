"""Scatter contiguous K and V tensors into physical non contingous blocks for efficient faster memory accessing"""
import torch 
import triton 
import triton.language as tl

from vllm.core.cache import BlockKCache

"""Load the K and V projections from contiguous memory, get the slot mapping for 
    each token position, store the projections in the cache """


@triton.jit
def store_kvcache_kernel(key_ptr, value_ptr, 
                        k_cache_ptr, v_cache_ptr, 
                        slot_mapping_ptr, 
                        stride_kn, stride_kh, 
                        stride_vn, stride_vh,
                        stride_kb_c, stride_ks_c, stride_kh_c,
                        stride_vb_c, stride_vs_c, stride_vh_c, 
                        BLOCK_SIZE:tl.constexpr, HEAD_DIM:tl.constexpr,):
   
   token_idx = tl.program_id(0) # each pid handles one token from the flattened batch
   slot_id = tl.load(slot_mapping_ptr + token_idx)

   if slot_id < 0:
      return 
   
   physical_block_id = slot_id // BLOCK_SIZE
   offset_in_block = slot_id % BLOCK_SIZE

   offset_dim = tl.arange(0, HEAD_DIM)
   current_head = tl.program_id(1)
   num_kv_head = tl.num_programs(1)

    # Get the pointer address to load from
   k_src_ptr = key_ptr + token_idx * stride_kn + current_head * stride_kh + offset_dim
   v_src_ptr = value_ptr + token_idx * stride_vn + current_head * stride_vh + offset_dim

   # Load from the fresh llama projection
   k_val = tl.load(k_src_ptr)
   v_val = tl.load(v_src_ptr)

   # Calculate the destination pointers in the paged cache to save the projection
   k_dst = k_cache_ptr + physical_block_id * stride_kb_c + offset_in_block * stride_ks_c + current_head * stride_kh_c + offset_dim
   v_dst = v_cache_ptr + physical_block_id * stride_vb_c + offset_in_block * stride_vs_c + current_head * stride_vh_c + offset_dim
   
   # store the projection in the cache
   tl.store(k_dst, k_val)
   tl.store(v_dst, v_val)
   

def store_kvcache(layer_idx, key, value, block_kv_cache, slot_mapping):
   k_cache_slice, v_cache_slice = block_kv_cache.get_layer_caches(layer_idx) #[num_blocks, block_size, num_kv_heads, head_dim]

   num_tokens, num_kv_heads, head_dim = key.shape
   D = num_kv_heads * head_dim

   assert key.stride(-1) == 1 and value.stride(-1) == 1
   assert key.stride(1) == head_dim and value.stride(1) == head_dim
   assert k_cache_slice.stride(1) == D and v_cache_slice.stride(1) == D
   assert slot_mapping.numel() ==  num_tokens

   grid = (num_tokens, num_kv_heads)
   store_kvcache_kernel[grid](key, value, k_cache_slice, v_cache_slice, slot_mapping, 
                              key.stride(0), key.stride(1),
                              value.stride(0), value.stride(1),
                              k_cache_slice.stride(0), k_cache_slice.stride(1), k_cache_slice.stride(2),
                              v_cache_slice.stride(0), v_cache_slice.stride(1), v_cache_slice.stride(2), 
                              BLOCK_SIZE=block_kv_cache.block_size,
                              HEAD_DIM=head_dim)