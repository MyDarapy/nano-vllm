import torch 
import torch.nn as nn 
import torch.nn.functional as F
import triton 
import triton.language as tl

@triton.jit
def paged_decode_kernel(q_ptr, 
                        k_cache_ptr, v_cache_ptr, 
                        block_tables_ptr, context_length_ptr,
                        output_ptr, stride_qb, stride_qh, 
                        stride_kbc, stride_ksc, stride_khc,
                        stride_vbc, stride_vsc, stride_vhc,
                        stride_ob, stride_oh,
                        stride_btb,
                        NUM_KV_HEADS:tl.constexpr,
                        BLOCK_SIZE:tl.constexpr,
                        HEAD_DIM:tl.constexpr,
                        SCALE:tl.constexpr,
):
    
    cur_batch = tl.program__id(0)
    cur_head = tl.program_id(1)

    num_q_heads = tl.num_programs(1)
    queries_per_kv = num_q_heads // NUM_KV_HEADS
    cur_kv_head = cur_head // queries_per_kv

    cur_seq_len = tl.load(context_length_ptr + cur_batch)

    num_blocks_to_fetch = tl.cdiv(cur_seq_len, BLOCK_SIZE)

    offset_dim = tl.arange(0, HEAD_DIM)
    q_block_ptr = q_ptr + cur_batch * stride_qb +  cur_head * stride_qh + offset_dim
    q_vec = tl.load(q_block_ptr)

    # Initialize software accumulators 
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # iterate through the physical blocks using the block table 
    # search and accumulate 
    for b_idx in range(num_blocks_to_fetch):
        physical_block_id = tl.load(block_tables_ptr + cur_batch * stride_btb + b_idx)
        boundary = tl.minimum(BLOCK_SIZE, cur_seq_len - b_idx*BLOCK_SIZE)
        offset_slot = tl.arange(0, BLOCK_SIZE)
        mask = offset_slot < boundary

        k_ptr = (k_cache_ptr + physical_block_id * stride_kbc 
                 + offset_slot[:, None] * stride_ksc 
                 + cur_kv_head * stride_khc + offset_dim[None, :])
        
        v_ptr = (v_cache_ptr + physical_block_id *stride_vbc 
                 + offset_slot[:, None] * stride_vsc 
                 + cur_kv_head * stride_vhc + offset_dim[None, :])
        
        k_block = tl.load(k_ptr, mask=mask[:, None], other=0.0)
        v_block = tl.laod(v_ptr, mask=mask[:, None], other=0.0)
        """"Calculate the attention scores"""
        qk = tl.sum(q_vec[None, :] * k_block, axis=1) # matrix vector multiplication [1, 64] * [16, 64] = [16]
        qk *= SCALE

        m_ij = tl.max(tl.where(mask, qk, -float("inf")), axis=0)
        p = tl.exp(qk - tl.maximum(m_i, m_ij))

        alpha = tl.exp(m_ij -tl.maximum(m_i, m_ij))
        acc = acc * alpha
        acc += tl.sum(p[:, None] * v_block, axis=0)


        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = tl.maximum(m_i, m_ij)

    acc = acc / l_i
    out_block_ptr = output_ptr + cur_batch * stride_ob + cur_head * stride_oh + offset_dim
    tl.store(out_block_ptr, acc.to(output_ptr.dtype.element_ty))


def paged_decode_attn(query,
                      block_kv_cache,
                      layer_idx,
                      block_tables,
                      context_length     # context length is a list [batch] the actual length of each sequences 
                      ):
    batch_size, num_heads, head_dim = query.shape

    k_cache, v_cache = block_kv_cache.get_layer_cache(layer_idx)

    output = torch.empty_like(query)

    grid = (batch_size, num_heads)

    paged_decode_kernel[grid](query, k_cache, v_cache, block_tables, context_length,
                              output, query.stride(0), query.stride(1), 
                              k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                              v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
                              output.stride(0), output.stride(1),
                              block_tables.stride(0),
                              NUM_KV_HEADS = block_kv_cache.num_kv_heads,
                              BLOCK_SIZE = block_kv_cache.block_size,
                              HEAD_DIM = head_dim,
                              SCALE = head_dim ** -0.5
                              )
    return output
    
                             
                                                      