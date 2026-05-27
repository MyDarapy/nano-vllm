from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _ensure_repo_root_on_path() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload-json", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--use-paged", action="store_true", default=True)
    ap.add_argument("--no-use-paged", dest="use_paged", action="store_false")
    ap.add_argument("--max-prefill-tokens", type=int, default=512)
    ap.add_argument("--enable-prefix-caching", action="store_true", default=True)
    ap.add_argument("--no-enable-prefix-caching", dest="enable_prefix_caching", action="store_false")
    ap.add_argument("--use-flash-attn", action="store_true", default=True)
    ap.add_argument("--no-use-flash-attn", dest="use_flash_attn", action="store_false")
    ap.add_argument("--warmup-steps", type=int, default=1)
    args = ap.parse_args()

    _ensure_repo_root_on_path()

    import torch  # type: ignore

    from vllm.engine import LLMEngine  # type: ignore
    from vllm.core.scheduler import SchedulingPolicy  # type: ignore

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    engine = LLMEngine(
        model_path=args.model,
        device=args.device,
        dtype=dtype_map[args.dtype],
        max_seq_len=args.max_model_len,
        max_batch_size=args.max_batch_size,
        use_paged_attention=args.use_paged,
        max_prefill_tokens=args.max_prefill_tokens,
        enable_prefix_caching=args.enable_prefix_caching,
        use_flash_attn=args.use_flash_attn,
        scheduling_policy=SchedulingPolicy.PRIORITY,
    )

    with open(args.workload_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    prompt_token_ids = payload["prompt_token_ids"]
    max_tokens_list = payload["max_tokens_list"]

    # Warmup: trigger compilation/autotune.
    warmup_prompt_len = max((len(p) for p in prompt_token_ids), default=4)
    warmup_ids = [1] * warmup_prompt_len
    for _ in range(args.warmup_steps):
        engine.scheduler.add_request(warmup_ids, max_tokens=1, priority=0)
    while engine.scheduler.has_pending_requests():
        engine.step()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for ids, mx in zip(prompt_token_ids, max_tokens_list):
        engine.scheduler.add_request(ids, max_tokens=mx, priority=0)
    while engine.scheduler.has_pending_requests():
        engine.step()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time() - t0

    total_requested = int(sum(max_tokens_list))
    print(json.dumps({"time_s": float(t), "output_tokens": total_requested}))


if __name__ == "__main__":
    main()
