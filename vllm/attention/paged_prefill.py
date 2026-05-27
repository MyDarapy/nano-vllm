import torch
import triton.language as tl
import math
import pdb 
import triton 

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    block_index_q,
    scale: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    off_kv: tl.constexpr,
    off_q: tl.constexpr,
    off_head: tl.constexpr,
    kn_stride: tl.constexpr,
    kd_stride: tl.constexpr,
    vd_stride: tl.constexpr,
    vn_stride: tl.constexpr,
    k_ptr,
    v_ptr,
    qkv_offset_K: tl.constexpr,
    qkv_offset_V: tl.constexpr,
    cur_len,
    HEAD_DIM: tl.constexpr):
    

    if STAGE == 1:
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else: 
        lo, hi = 0, cur_len 
    for start_kv in range (lo, hi, BLOCK_SIZE_KV):
        kv_positions = start_kv + off_kv
        K_block_ptr = k_ptr + qkv_offset_K + off_head[:, None] * kd_stride + kv_positions[None, :] * kn_stride
        V_block_ptr = v_ptr + qkv_offset_V + kv_positions[:, None] * vn_stride + off_head[None, :] * vd_stride

        mask_k = kv_positions[None, :] < cur_len
        mask_v = kv_positions[:, None] < cur_len

        K_block = tl.load(K_block_ptr, mask=mask_k, other=0.0)
        V_block = tl.load(V_block_ptr, mask=mask_v, other=0.0)   

        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            mask = off_q[:, None] >= (start_kv + off_kv[None, :])
            QK_block = QK_block * scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            m_ij= tl.maximum(m_i, tl.max(QK_block, 1) * scale)
            QK_block = QK_block * scale - m_ij[:, None]
        
        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, 1)
        alpha = tl.math.exp(m_i - m_ij) 
        l_i = l_i * alpha + l_ij

        P_block = P_block.to(tl.float16)
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij 
    return O_block, l_i, m_i 



config = [triton.Config({"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
                  num_stages = num_stages, num_warps=num_warps)
    for BLOCK_SIZE_Q in [128, 256, 512]
    for BLOCK_SIZE_KV in [32, 64, 128]
    for num_stages in [1, 2, 3, 4]
    for num_warps in [2, 4, 8]]


@triton.autotune(config, key = ["SEQ_LEN", "HEAD_DIM"])


@triton.jit
def fwd_flash_attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, m_ptr, context_len_ptr, scale,
                          qb_stride, qh_stride, qn_stride, qd_stride,
                          kb_stride, kh_stride, kn_stride, kd_stride,
                          vb_stride, vh_stride, vn_stride, vd_stride,
                          ob_stride, oh_stride, on_stride, od_stride,
                          BATCH_SIZE, NUM_HEADS:tl.constexpr, NUM_KV_HEADS:tl.constexpr, SEQ_LEN:tl.constexpr, HEAD_DIM:tl.constexpr, 
                          BLOCK_SIZE_Q:tl.constexpr, BLOCK_SIZE_KV:tl.constexpr, STAGE:tl.constexpr):
    
    
    # get the id of this program instance
    block_index_q = tl.program_id(0) # Which chunk of sequence this program is responsible for. (Chunk of sequence within the tile)
    index_batch_head = tl.program_id(1) # what batch-head to process. zooms out

    # get exact batch 
    index_batch = index_batch_head // NUM_HEADS

    # get exact head 
    index_head = index_batch_head % NUM_HEADS

    # GQA FIX
    queries_per_kv_head = NUM_HEADS // NUM_KV_HEADS
    index_kv_head = index_head // queries_per_kv_head

    cur_len = tl.load(context_len_ptr + index_batch)
    # create offsets to get the index of sequences we are going to process
    qkv_offset = index_batch * qb_stride + index_head * qh_stride # i.e move from the first to the correct batch then move to the correct head within that batch 
    qkv_offset_K = index_batch * kb_stride + index_kv_head * kh_stride
    qkv_offset_V = index_batch * vb_stride + index_kv_head * vh_stride
    qkv_offset_O = index_batch * ob_stride + index_head * oh_stride

    off_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q) # same as off_q (in this head what q block do we need to read )
    off_kv = tl.arange(0, BLOCK_SIZE_KV)
    off_head = tl.arange(0, HEAD_DIM)

    # create blocks of pointers to get the address of where the index lives 
    Q_block_ptr = q_ptr + qkv_offset + off_q[:, None] * qn_stride + off_head[None, :] * qd_stride
    O_block_ptr = o_ptr + qkv_offset_O + off_q[:, None] * on_stride + off_head[None, :] * od_stride
    q_mask_1d = off_q < cur_len
    q_mask_2d = q_mask_1d[:, None] 

    m_i = tl.zeros((BLOCK_SIZE_Q,), dtype= tl.float32) - float("inf")

    l_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32) + 1.0
    O_block = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)
    Q_block = tl.load(Q_block_ptr, mask=q_mask_2d, other=0.0) # add a mask

    # stage 1: Blocks before the diagonal 
    # stage 2: diagonal block itself 
    # stage 3: for non-causal no masking is needed. For causal mask all the blocks here.
    
    # runs if causal is True i.e we mask out the future tokens from contributing
    # this if statement executes for non-causal attention (no masking) or for the blocks to the left of the diagonal in the causal attention
    # Stage = 3 if causal else 1 
    if STAGE == 1 or STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i, 
            Q_block, 
            block_index_q,
            scale, 
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV, 
            4 - STAGE,
            off_kv,
            off_q,
            off_head,
            kn_stride,
            kd_stride,
            vd_stride,
            vn_stride, 
            k_ptr,
            v_ptr,
            qkv_offset_K,
            qkv_offset_V,
            cur_len, 
            HEAD_DIM
        )
    
    # this executes for blocks to the right of the diagonal in the causal attention
    if STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i, 
            Q_block, 
            block_index_q,
            scale, 
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV, 
            2,
            off_kv,
            off_q,
            off_head,
            kn_stride,
            kd_stride,
            vd_stride,
            vn_stride, 
            k_ptr,
            v_ptr,
            qkv_offset_K,
            qkv_offset_V,
            cur_len, 
            HEAD_DIM
        )

    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = m_ptr + index_batch_head * SEQ_LEN + off_q 
    tl.store(m_ptrs, m_i, mask=q_mask_1d)
    tl.store(O_block_ptr, O_block.to(tl.float16), mask=q_mask_2d)


# Host wrapper that prepares our inputs and parameters and runs the triton kernel
class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def flash_attention(Q, K, V, context_lens, causal):
        assert Q.is_cuda
        assert K.is_cuda
        assert V.is_cuda

        B, Hq, Lq, D = Q.shape
        B, Hk, Lk, D = K.shape
        B, Hk, Lk, D = V.shape

        
        # create the output buffer
        O = torch.empty_like(Q)

        #we set block_sizes manually for now. We will autotune this later
        BLOCK_SIZE_Q = 128
        BLOCK_SIZE_KV = 32

        
        stage = 3 if causal else 1

        grid = lambda x: (triton.cdiv(Lq, x["BLOCK_SIZE_Q"]),
                          B * Hq, 1)
        M = torch.empty((B, Hq, Lq), device=Q.device, dtype=torch.float32)

        scaling_factor = 1 / math.sqrt(D)
        fwd_flash_attn_kernel[grid](Q, K, V, O, M, context_lens, scaling_factor,
                                    Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                                    K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                                    V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                                    O.stride(0), O.stride(1), O.stride(2), O.stride(3),
                                    B, NUM_HEADS=Hq, NUM_KV_HEADS=Hk, SEQ_LEN=Lq, HEAD_DIM=D,STAGE=stage,)
        #ctx.save_for_backward
    
        return O


def testing(BATCH_SIZE, NUM_HEADS, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    
    # standard attention 
    softmax_scale = 1/(HEAD_DIM ** 0.5)
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    reference_O = torch.matmul(P, V)


    # triton 
    tri_out = TritonFlashAttention.flash_attention(Q, K, V, causal).half()

    rtol =0.0
    atol =1e-2
    assert torch.allclose(reference_O, tri_out, atol=atol, rtol=rtol)

if __name__ == "__main__":
    testing(BATCH_SIZE=8, NUM_HEADS=16, NUM_KV_HEADS=4, SEQ_LEN=512, HEAD_DIM=64, causal=True)
    print("causal worked!")
    testing(BATCH_SIZE=8, NUM_HEADS=16, NUM_KV_HEADS=4, SEQ_LEN=512, HEAD_DIM=64, causal=False)
    print("Success! non causal worked")