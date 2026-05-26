#!/usr/bin/env python3
"""
Benchmark nano-vllm engine latency/throughput.

Metrics reported:
  - TTFT (time-to-first-token): wall time from request submission to first output token
  - ITL (inter-token latency): per-request latency between output tokens after the first
  - Throughput: generated output tokens / second (aggregate)

This script avoids modifying the engine by observing Sequence objects in the scheduler.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _now() -> float:
    return time.perf_counter()


def _maybe_cuda_sync(device: str) -> None:
    # Keep import local so the file can be imported without torch installed.
    try:
        import torch  # type: ignore
    except Exception:
        return
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _format_ms(x_s: float) -> str:
    return f"{x_s * 1e3:.2f} ms"


@dataclass
class SeqStats:
    submit_t: float
    prompt_len: Optional[int] = None
    max_tokens: Optional[int] = None
    token_ts: List[float] = field(default_factory=list)  # output token timestamps

    @property
    def ttft_s(self) -> Optional[float]:
        if not self.token_ts:
            return None
        return self.token_ts[0] - self.submit_t

    @property
    def itl_s_list(self) -> List[float]:
        if len(self.token_ts) < 2:
            return []
        return [self.token_ts[i] - self.token_ts[i - 1] for i in range(1, len(self.token_ts))]


def _collect_all_sequences(engine) -> List[object]:
    # scheduler.waiting() returns plain Sequence objects (unlike _waiting_heap).
    seqs = []
    try:
        seqs.extend(engine.scheduler.waiting())
    except Exception:
        pass
    try:
        seqs.extend(engine.scheduler.running)
    except Exception:
        pass
    try:
        seqs.extend(engine.scheduler.finished)
    except Exception:
        pass
    # Dedup by seq_id if present.
    out = {}
    for s in seqs:
        sid = getattr(s, "seq_id", id(s))
        out[sid] = s
    return list(out.values())


def _make_prompts(num_requests: int, prompt_chars: int) -> List[str]:
    # Simple, deterministic prompt with controllable rough length.
    base = "Write one sentence about benchmarking LLM inference latency. "
    if prompt_chars <= len(base):
        return [base[:prompt_chars] for _ in range(num_requests)]
    pad = "x" * (prompt_chars - len(base))
    return [base + pad for _ in range(num_requests)]


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    values_sorted = sorted(values)
    n = len(values_sorted)

    def pct(p: float) -> float:
        if n == 1:
            return values_sorted[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return values_sorted[f]
        return values_sorted[f] + (values_sorted[c] - values_sorted[f]) * (k - f)

    return {
        "count": float(n),
        "mean": statistics.mean(values_sorted),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "min": values_sorted[0],
        "max": values_sorted[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local model path")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--use-paged-attention", action="store_true", default=True)
    ap.add_argument("--no-use-paged-attention", dest="use_paged_attention", action="store_false")
    ap.add_argument("--use-flash-attn", action="store_true", default=True)
    ap.add_argument("--no-use-flash-attn", dest="use_flash_attn", action="store_false")
    ap.add_argument("--max-prefill-tokens", type=int, default=512)
    ap.add_argument("--enable-prefix-caching", action="store_true", default=True)
    ap.add_argument("--no-enable-prefix-caching", dest="enable_prefix_caching", action="store_false")
    ap.add_argument("--num-requests", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--prompt-chars", type=int, default=256, help="Approx prompt size (chars)")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup requests (not counted)")
    ap.add_argument(
        "--exclude-tokenization",
        action="store_true",
        help="Measure TTFT/ITL starting after tokenization (closer to pure engine latency).",
    )
    args = ap.parse_args()

    # local import so tooling that inspects this file doesn't require torch/transformers.
    import torch  # type: ignore
    from vllm.engine import LLMEngine  # type: ignore
    from vllm.core.scheduler import SchedulingPolicy  # type: ignore

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    engine = LLMEngine(
        model_path=args.model,
        device=args.device,
        dtype=dtype_map[args.dtype],
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        use_paged_attention=args.use_paged_attention,
        max_prefill_tokens=args.max_prefill_tokens,
        enable_prefix_caching=args.enable_prefix_caching,
        use_flash_attn=args.use_flash_attn,
        scheduling_policy=SchedulingPolicy.PRIORITY,
    )

    # Warmup (optional): minimal requests to load kernels / stabilize clocks.
    if args.warmup > 0:
        warm_prompts = _make_prompts(args.warmup, min(args.prompt_chars, 128))
        for p in warm_prompts:
            engine.add_request(p, max_tokens=4, priority=0)
        while engine.scheduler.has_pending_requests():
            _maybe_cuda_sync(args.device)
            engine.step()
            _maybe_cuda_sync(args.device)

    prompts = _make_prompts(args.num_requests, args.prompt_chars)

    seq_stats: Dict[int, SeqStats] = {}
    prev_out_lens: Dict[int, int] = {}
    total_step_s = 0.0
    total_new_tokens = 0

    t_start = _now()
    for p in prompts:
        if args.exclude_tokenization:
            messages = [{"role": "user", "content": p}]
            prompt_token_ids = engine.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            t_submit = _now()
            seq = engine.scheduler.add_request(
                prompt_token_ids,
                max_tokens=args.max_new_tokens,
                priority=0,
            )
            seq_id = seq.seq_id
            engine._prompts[seq_id] = p
        else:
            t_submit = _now()
            seq_id = engine.add_request(p, max_tokens=args.max_new_tokens, priority=0)
        seq_stats[seq_id] = SeqStats(submit_t=t_submit, max_tokens=args.max_new_tokens)
        prev_out_lens[seq_id] = 0

    # Main loop: step until all are done, recording token timestamps.
    while engine.scheduler.has_pending_requests():
        _maybe_cuda_sync(args.device)
        t0 = _now()
        engine.step()
        _maybe_cuda_sync(args.device)
        t1 = _now()
        dt = t1 - t0
        total_step_s += dt

        # Observe sequences and check for newly appended output tokens.
        for seq in _collect_all_sequences(engine):
            sid = getattr(seq, "seq_id", None)
            if sid is None or sid not in seq_stats:
                continue

            if seq_stats[sid].prompt_len is None:
                try:
                    seq_stats[sid].prompt_len = seq.get_prompt_len()
                except Exception:
                    pass

            try:
                out_len = seq.get_output_len()
            except Exception:
                continue

            prev_len = prev_out_lens.get(sid, 0)
            if out_len > prev_len:
                # In this engine, output_len increments by 1 per step.
                delta = out_len - prev_len
                seq_stats[sid].token_ts.extend([t1] * delta)
                total_new_tokens += delta
                prev_out_lens[sid] = out_len

    t_end = _now()

    # Aggregate metrics.
    ttft = [s.ttft_s for s in seq_stats.values() if s.ttft_s is not None]
    itl_all = []
    for s in seq_stats.values():
        itl_all.extend(s.itl_s_list)

    ttft_sum = _summarize([x for x in ttft if x is not None])
    itl_sum = _summarize(itl_all)

    wall_s = t_end - t_start
    throughput_toks_s = (total_new_tokens / wall_s) if wall_s > 0 else 0.0

    print("=== nano-vllm benchmark ===")
    print(f"requests: {args.num_requests} | max_new_tokens/request: {args.max_new_tokens}")
    print(f"device: {args.device} | dtype: {args.dtype}")
    print(f"wall_time: {wall_s:.3f} s | step_time_sum: {total_step_s:.3f} s")
    print(f"generated_tokens: {total_new_tokens} | throughput: {throughput_toks_s:.2f} tok/s")

    if ttft_sum:
        print("--- TTFT (prefill latency proxy) ---")
        print(
            f"count={int(ttft_sum['count'])} mean={_format_ms(ttft_sum['mean'])} "
            f"p50={_format_ms(ttft_sum['p50'])} p90={_format_ms(ttft_sum['p90'])} "
            f"p95={_format_ms(ttft_sum['p95'])} max={_format_ms(ttft_sum['max'])}"
        )
    else:
        print("--- TTFT ---")
        print("No TTFT samples (no tokens generated?)")

    if itl_sum:
        print("--- ITL (inter-token latency) ---")
        print(
            f"count={int(itl_sum['count'])} mean={_format_ms(itl_sum['mean'])} "
            f"p50={_format_ms(itl_sum['p50'])} p90={_format_ms(itl_sum['p90'])} "
            f"p95={_format_ms(itl_sum['p95'])} max={_format_ms(itl_sum['max'])}"
        )
    else:
        print("--- ITL ---")
        print("No ITL samples (each request produced <2 tokens)")


if __name__ == "__main__":
    main()
