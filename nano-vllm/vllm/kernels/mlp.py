import triton 
import torch
import triton.language as tl


@triton.jit 
def ffn_stage_2(hidden_ptr, w_down_ptr, output_ptr, B, T, C, N,
               stride_hb, stride_ht, stride_hn, w_down_n, w_down_c,
               ob, ot, oc,
               BLOCK_SIZE_BT:tl.constexpr, BLOCK_SIZE_C:tl.constexpr, BLOCK_SIZE_N:tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_block_start = pid_m * BLOCK_SIZE_BT
    n_block_start = pid_n * BLOCK_SIZE_C

    off_h_bt = m_block_start + tl.arange(0, BLOCK_SIZE_BT)
    off_w_down = n_block_start + tl.arange(0, BLOCK_SIZE_C)
    off_n = tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_BT, BLOCK_SIZE_C), dtype=tl.float32)
    
    for n_idx in range(0, N, BLOCK_SIZE_N):
        col_n = n_idx + tl.arange(0, BLOCK_SIZE_N)

        off_h_bt = m_block_start + tl.arange(0, BLOCK_SIZE_BT)
        off_w_down = n_block_start + tl.arange(0, BLOCK_SIZE_C)


        hidden_pointer = hidden_ptr + off_h_bt[:, None] * stride_ht + col_n[None, :] * stride_hn
        down_pointer = w_down_ptr + col_n[None, :] * w_down_n + off_w_down[None, :] * w_down_c

        row_mask = off_h_bt < (B * T)
        n_mask = col_n < N
        c_mask = off_w_down < C

        hidden = tl.load(hidden_pointer, mask= row_mask[:, None] & n_mask[None, :], other=0.0)
        down = tl.load(down_pointer, mask=n_mask[:, None] & c_mask[None, :], other=0.0)
        acc = tl.dot(hidden, down, acc)
        
    acc = acc.to(tl.float16)

    off_m = m_block_start + tl.arange(0, BLOCK_SIZE_BT)
    off_n = n_block_start + tl.arange(0, BLOCK_SIZE_C)
    output_pointer = output_ptr + off_m[:, None] * ot + off_n[None, :] * oc
    output = tl.store(output_pointer, acc, mask=row_mask[:, None] & c_mask[None, :])

    return output
    
config = [triton.Config({"BLOCK_SIZE_BT": BLOCK_SIZE_BT, "BLOCK_SIZE_N": BLOCK_SIZE_N, "BLOCK_SIZE_C": BLOCK_SIZE_C},
                  num_stages = num_stages, num_warps=num_warps)
    for BLOCK_SIZE_BT in [128, 256, 512]
    for BLOCK_SIZE_N in [128, 64, 32]
    for BLOCK_SIZE_C in [32, 64, 128]
    for num_stages in [1, 2, 3, 4]
    for num_warps in [2, 4, 8]]


@triton.autotune(config, key = ["BLOCK_SIZE_BT", "BLOCK_SIZE_N", "BLOCK_SIZE_C"])
    
@triton.jit
def ffn_kernel(x_ptr, w_gate_ptr, w_up_ptr, output_ptr, B, T, C, N,
               stride_xb, stride_xt, stride_xc, w_gate_c, w_gate_n, w_up_c, w_up_n,
               ob, ot, on,
               BLOCK_SIZE_BT:tl.constexpr, BLOCK_SIZE_C:tl.constexpr, BLOCK_SIZE_N:tl.constexpr):
    
    # Get the index for this program instance that tells it what this particular instance needs to work on.
    
    index_batch_token = tl.program_id(0) # tells me which group of B*T rows in the output matrix this PI handles
    index_intermidate_dim = tl.program_id(1) # tells me which group of N cols in the output matrix this PI handles

    # compute where to load this PI data blocks from in the input matrix 
    m_block_start = index_batch_token * BLOCK_SIZE_BT
    n_block_start = index_intermidate_dim * BLOCK_SIZE_N 

    # create offset
    off_x_bt = m_block_start + tl.arange(0, BLOCK_SIZE_BT)
    off_w_gate = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    off_w_up = n_block_start + tl.arange(0, BLOCK_SIZE_N)
   
    acc_x_gate = tl.zeros((BLOCK_SIZE_BT, BLOCK_SIZE_N), dtype=tl.float32)
    acc_x_up = tl.zeros((BLOCK_SIZE_BT, BLOCK_SIZE_N),  dtype=tl.float32)

    # gate and up individual matrix multiplication 
    for k_idx in range(0, C, BLOCK_SIZE_C):
        col_k = k_idx + tl.arange(0, BLOCK_SIZE_C)

        # load from global mem
        x_pointers = x_ptr + off_x_bt[:, None] * stride_xt + col_k[None, :] * stride_xc
        w_gate_pointers = w_gate_ptr + col_k[:, None] * w_gate_c + off_w_gate[None, :] * w_gate_n
        w_up_pointers = w_up_ptr + col_k[:, None] * w_up_c + off_w_up[None, :] * w_up_n
        
        row_mask = off_x_bt < (B*T)
        k_mask = col_k < C
        n_mask = off_w_up < N 


        x = tl.load(x_pointers, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        w_gate = tl.load(w_gate_pointers, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        w_up = tl.load(w_up_pointers, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        acc_x_gate = tl.dot(x, w_gate, acc_x_gate)
        acc_x_up = tl.dot(x, w_up, acc_x_up)
    
    gate = acc_x_gate.to(tl.float16)
    up = acc_x_up.to(tl.float16)
    hidden_tile = (gate * (1 / (1 + tl.exp(-gate)))) * up

    off_m = m_block_start + tl.arange(0, BLOCK_SIZE_BT)
    off_n = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    output_pointers = output_ptr + off_m[:, None] * ot + off_n[None, :] * on
    output = tl.store(output_pointers, hidden_tile, mask=row_mask[:, None] & n_mask[None, :])
    return output


class MLP(torch.autograd.Function):
    @staticmethod
    def ffn_stage_1(x, w_gate, w_up):
        assert x.is_cuda()
        assert w_gate.is_cuda()
        assert w_up.is_cuda()

        B, T, C = x.shape
        C, N = w_gate.shape
        
        BLOCK_SIZE_BT = 64
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_C = 4 # the reduction dimension
        
        output = torch.empty((B, T, N))
        
        grid = (triton.cdiv(B*T, BLOCK_SIZE_BT), triton.cdiv(N, BLOCK_SIZE_N))
                
        ffn_kernel[grid](
            x, w_gate, w_up, output, B, T, C, N,
            x.stride(0), x.stride(1), x.stride(2), 
            w_gate.stride(0), w_gate.stride(1), w_up.stride(0), w_up.stride(1), 
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_SIZE_BT, BLOCK_SIZE_C, BLOCK_SIZE_N)
        return output
    
    def ffn_stage_2(hidden_state, w_down):
        assert hidden_state.is_cuda()
        assert w_down.is_cuda()

        B, T, N = hidden_state.shape
        N, C = w_down.shape

        BLOCK_SIZE_BT = 64
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_C = 4

        output = torch.empty((B, T, C))
        grid = (triton.cdiv(B*T, BLOCK_SIZE_BT), triton.cdiv(C, BLOCK_SIZE_C))

        ffn_stage_2[grid](
            hidden_state, w_down, output, B, T, C, N,
            hidden_state.stride(0), hidden_state.stride(1), hidden_state.stride(2), 
            w_down.stride(0), w_down.stride(1), 
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_SIZE_BT, BLOCK_SIZE_C, BLOCK_SIZE_N)
        return output

