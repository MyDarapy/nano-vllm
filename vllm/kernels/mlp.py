import torch
import triton
import triton.language as tl


_STAGE1_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_BT": block_bt,
            "BLOCK_SIZE_C": block_c,
            "BLOCK_SIZE_N": block_n,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_bt in [32, 64, 128]
    for block_c in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [1, 2, 3, 4]
    for num_warps in [2, 4, 8]
]


_STAGE2_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_BT": block_bt,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_C": block_c,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_bt in [32, 64, 128]
    for block_n in [32, 64, 128]
    for block_c in [32, 64, 128]
    for num_stages in [1, 2, 3, 4]
    for num_warps in [2, 4, 8]
]


@triton.autotune(configs=_STAGE1_CONFIGS, key=["B", "T", "C", "N"])
@triton.jit
def _ffn_stage_1_kernel(
    x_ptr,
    w_gate_ptr,
    w_up_ptr,
    output_ptr,
    B,
    T,
    C,
    N,
    stride_xb,
    stride_xt,
    stride_xc,
    stride_wg_c,
    stride_wg_n,
    stride_wu_c,
    stride_wu_n,
    stride_ob,
    stride_ot,
    stride_on,
    BLOCK_SIZE_BT: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_block_start = pid_m * BLOCK_SIZE_BT
    n_block_start = pid_n * BLOCK_SIZE_N

    off_bt = m_block_start + tl.arange(0, BLOCK_SIZE_BT)
    off_b = off_bt // T
    off_t = off_bt % T
    off_n = n_block_start + tl.arange(0, BLOCK_SIZE_N)

    acc_gate = tl.zeros((BLOCK_SIZE_BT, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_BT, BLOCK_SIZE_N), dtype=tl.float32)

    row_mask = off_bt < (B * T)
    n_mask = off_n < N

    for k_start in range(0, C, BLOCK_SIZE_C):
        off_k = k_start + tl.arange(0, BLOCK_SIZE_C)
        k_mask = off_k < C

        x_ptrs = (
            x_ptr
            + off_b[:, None] * stride_xb
            + off_t[:, None] * stride_xt
            + off_k[None, :] * stride_xc
        )
        w_gate_ptrs = (
            w_gate_ptr
            + off_k[:, None] * stride_wg_c
            + off_n[None, :] * stride_wg_n
        )
        w_up_ptrs = (
            w_up_ptr
            + off_k[:, None] * stride_wu_c
            + off_n[None, :] * stride_wu_n
        )

        x = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        w_gate = tl.load(w_gate_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        w_up = tl.load(w_up_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc_gate = tl.dot(x, w_gate, acc_gate)
        acc_up = tl.dot(x, w_up, acc_up)

    gate = acc_gate
    up = acc_up
    silu_gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
    hidden = (silu_gate * up).to(tl.float16)

    out_ptrs = (
        output_ptr
        + off_b[:, None] * stride_ob
        + off_t[:, None] * stride_ot
        + off_n[None, :] * stride_on
    )
    tl.store(out_ptrs, hidden, mask=row_mask[:, None] & n_mask[None, :])


@triton.autotune(configs=_STAGE2_CONFIGS, key=["B", "T", "N", "C"])
@triton.jit
def _ffn_stage_2_kernel(
    hidden_ptr,
    w_down_ptr,
    output_ptr,
    B,
    T,
    N,
    C,
    stride_hb,
    stride_ht,
    stride_hn,
    stride_wd_n,
    stride_wd_c,
    stride_ob,
    stride_ot,
    stride_oc,
    BLOCK_SIZE_BT: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    m_block_start = pid_m * BLOCK_SIZE_BT
    c_block_start = pid_c * BLOCK_SIZE_C

    off_bt = m_block_start + tl.arange(0, BLOCK_SIZE_BT)
    off_b = off_bt // T
    off_t = off_bt % T
    off_c = c_block_start + tl.arange(0, BLOCK_SIZE_C)

    acc = tl.zeros((BLOCK_SIZE_BT, BLOCK_SIZE_C), dtype=tl.float32)

    row_mask = off_bt < (B * T)
    c_mask = off_c < C

    for n_start in range(0, N, BLOCK_SIZE_N):
        off_n = n_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = off_n < N

        hidden_ptrs = (
            hidden_ptr
            + off_b[:, None] * stride_hb
            + off_t[:, None] * stride_ht
            + off_n[None, :] * stride_hn
        )
        w_down_ptrs = (
            w_down_ptr
            + off_n[:, None] * stride_wd_n
            + off_c[None, :] * stride_wd_c
        )

        hidden = tl.load(hidden_ptrs, mask=row_mask[:, None] & n_mask[None, :], other=0.0)
        w_down = tl.load(w_down_ptrs, mask=n_mask[:, None] & c_mask[None, :], other=0.0)

        acc = tl.dot(hidden, w_down, acc)

    out = acc.to(tl.float16)
    out_ptrs = (
        output_ptr
        + off_b[:, None] * stride_ob
        + off_t[:, None] * stride_ot
        + off_c[None, :] * stride_oc
    )
    tl.store(out_ptrs, out, mask=row_mask[:, None] & c_mask[None, :])


class MLP:
    @staticmethod
    def ffn_stage_1(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda
        assert w_gate.is_cuda
        assert w_up.is_cuda

        x = x.contiguous()
        w_gate = w_gate.contiguous()
        w_up = w_up.contiguous()

        B, T, C = x.shape
        C_w, N = w_gate.shape
        assert C == C_w
        assert w_up.shape == (C, N)

        output = torch.empty((B, T, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(B * T, meta["BLOCK_SIZE_BT"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

        _ffn_stage_1_kernel[grid](
            x,
            w_gate,
            w_up,
            output,
            B,
            T,
            C,
            N,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            w_gate.stride(0),
            w_gate.stride(1),
            w_up.stride(0),
            w_up.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
        )
        return output

    @staticmethod
    def ffn_stage_2(hidden_state: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
        assert hidden_state.is_cuda
        assert w_down.is_cuda

        hidden_state = hidden_state.contiguous()
        w_down = w_down.contiguous()

        B, T, N = hidden_state.shape
        N_w, C = w_down.shape
        assert N == N_w

        output = torch.empty((B, T, C), device=hidden_state.device, dtype=hidden_state.dtype)

        grid = lambda meta: (
            triton.cdiv(B * T, meta["BLOCK_SIZE_BT"]),
            triton.cdiv(C, meta["BLOCK_SIZE_C"]),
        )

        _ffn_stage_2_kernel[grid](
            hidden_state,
            w_down,
            output,
            B,
            T,
            N,
            C,
            hidden_state.stride(0),
            hidden_state.stride(1),
            hidden_state.stride(2),
            w_down.stride(0),
            w_down.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
        )
        return output