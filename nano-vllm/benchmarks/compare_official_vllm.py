from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from random import randint, seed
from typing import Any, Dict, List, Tuple


@dataclass
class Result:
    name: str
    output_tokens: int
    time_s: float

    @property
    def throughput(self) -> float:
        return self.output_tokens / self.time_s if self.time_s > 0 else 0.0


def _repo_root() -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, ".."))


def _infer_vocab_size(model_path: str) -> int:
    # Use HF config if available; else fall back to a conservative guess.
    try:
        from transformers import AutoConfig  # type: ignore
    except Exception:
        return 32000
    try:
        cfg = AutoConfig.from_pretrained(model_path)
        v = int(getattr(cfg, "vocab_size"))
        return v if v > 0 else 32000
    except Exception:
        return 32000


def _build_workload(
    *,
    num_seqs: int,
    min_input_len: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
    vocab_size: int,
    max_model_len: int,
) -> Tuple[List[List[int]], List[int]]:
    prompts: List[List[int]] = []
    max_tokens_list: List[int] = []
    for _ in range(num_seqs):
        in_len = randint(min_input_len, max_input_len)
        out_len = randint(min_output_len, max_output_len)
        # Ensure prompt+output fits the model length budget.
        out_len = min(out_len, max(1, max_model_len - in_len))
        prompts.append([randint(0, vocab_size - 1) for _ in range(in_len)])
        max_tokens_list.append(out_len)
    return prompts, max_tokens_list


def _run_subprocess_json(python: str, argv: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    proc = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Subprocess benchmark failed.\n"
            f"python={python}\n"
            f"exit_code={proc.returncode}\n"
            "---- stdout ----\n"
            f"{proc.stdout}\n"
            "---- stderr ----\n"
            f"{proc.stderr}\n"
        )
    # JSON is printed as the last line.
    last = proc.stdout.strip().splitlines()[-1]
    return json.loads(last)


def _print_table(results: List[Result]) -> None:
    print()
    print("| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |")
    print("|---|---:|---:|---:|")
    for r in results:
        print(f"| {r.name} | {r.output_tokens:,} | {r.time_s:.2f} | {r.throughput:.2f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--nano-python", required=True, help="Python interpreter for nano env")
    ap.add_argument("--vllm-python", required=True, help="Python interpreter for vLLM env")
    ap.add_argument("--num-seqs", type=int, default=256)
    ap.add_argument("--min-input-len", type=int, default=100)
    ap.add_argument("--max-input-len", type=int, default=1024)
    ap.add_argument("--min-output-len", type=int, default=100)
    ap.add_argument("--max-output-len", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab-size", type=int, default=0, help="0=auto (HF config) else fixed")
    ap.add_argument("--max-batch-size", type=int, default=8, help="nano engine max_batch_size")
    ap.add_argument(
        "--max-prefill-tokens",
        type=int,
        default=8192,
        help="nano scheduler prefill token budget (set >= max_batch_size*max_input_len to avoid chunked prefill)",
    )
    ap.add_argument("--warmup-steps", type=int, default=1)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-paged", action="store_true", default=True)
    ap.add_argument("--no-use-paged", dest="use_paged", action="store_false")
    ap.add_argument("--use-flash-attn", action="store_true", default=True)
    ap.add_argument("--no-use-flash-attn", dest="use_flash_attn", action="store_false")
    ap.add_argument("--enable-prefix-caching", action="store_true", default=True)
    ap.add_argument("--no-enable-prefix-caching", dest="enable_prefix_caching", action="store_false")
    ap.add_argument(
        "--include-nano-legacy",
        action="store_true",
        default=True,
        help="Also benchmark nano-vLLM legacy (non-paged) path using the same workload.",
    )
    ap.add_argument("--no-include-nano-legacy", dest="include_nano_legacy", action="store_false")
    args = ap.parse_args()

    seed(args.seed)
    model_path = os.path.expanduser(args.model)

    vocab_size = args.vocab_size if args.vocab_size > 0 else _infer_vocab_size(model_path)

    prompt_token_ids, max_tokens_list = _build_workload(
        num_seqs=args.num_seqs,
        min_input_len=args.min_input_len,
        max_input_len=args.max_input_len,
        min_output_len=args.min_output_len,
        max_output_len=args.max_output_len,
        vocab_size=vocab_size,
        max_model_len=args.max_model_len,
    )

    with tempfile.TemporaryDirectory() as tmpd:
        workload_json = os.path.join(tmpd, "workload.json")
        with open(workload_json, "w") as f:
            json.dump({"prompt_token_ids": prompt_token_ids, "max_tokens_list": max_tokens_list}, f)

        base_env = dict(os.environ)

        # nano env: ensure repo root is on PYTHONPATH so it imports this repo's `vllm/`.
        nano_env = dict(base_env)
        nano_env["PYTHONPATH"] = _repo_root() + (os.pathsep + nano_env["PYTHONPATH"] if "PYTHONPATH" in nano_env else "")

        # vllm env: avoid importing this repo's `vllm/` by not adding repo root to PYTHONPATH.
        vllm_env = dict(base_env)
        vllm_env.pop("PYTHONPATH", None)
        # Force V0 so we can reliably disable chunked prefill for apples-to-apples comparisons.
        # (vLLM V1 effectively always enables chunked prefill.)
        vllm_env["VLLM_USE_V1"] = "0"

        nano_runner = os.path.join(_repo_root(), "benchmarks", "run_nano_engine.py")
        vllm_runner = os.path.join(_repo_root(), "benchmarks", "run_official_vllm.py")

        nano_argv = [
            args.nano_python,
            nano_runner,
            "--workload-json",
            workload_json,
            "--model",
            model_path,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--max-model-len",
            str(args.max_model_len),
            "--max-batch-size",
            str(args.max_batch_size),
            "--max-prefill-tokens",
            str(args.max_prefill_tokens),
            "--warmup-steps",
            str(args.warmup_steps),
        ]
        if not args.use_paged:
            nano_argv.append("--no-use-paged")
        if not args.use_flash_attn:
            nano_argv.append("--no-use-flash-attn")
        if not args.enable_prefix_caching:
            nano_argv.append("--no-enable-prefix-caching")

        nano_legacy_argv = None
        if args.include_nano_legacy:
            nano_legacy_argv = list(nano_argv)
            if "--no-use-paged" not in nano_legacy_argv:
                nano_legacy_argv.append("--no-use-paged")

        vllm_argv = [
            args.vllm_python,
            vllm_runner,
            "--workload-json",
            workload_json,
            "--model",
            model_path,
            "--max-model-len",
            str(args.max_model_len),
            "--warmup-steps",
            str(args.warmup_steps),
            "--disable-chunked-prefill",
        ]

        nano_out = _run_subprocess_json(args.nano_python, nano_argv, nano_env)
        nano_legacy_out = None
        if nano_legacy_argv is not None:
            nano_legacy_out = _run_subprocess_json(args.nano_python, nano_legacy_argv, nano_env)
        vllm_out = _run_subprocess_json(args.vllm_python, vllm_argv, vllm_env)

        results = [
            Result("vLLM", int(vllm_out["output_tokens"]), float(vllm_out["time_s"])),
            Result("Nano-vLLM (paged)", int(nano_out["output_tokens"]), float(nano_out["time_s"])),
        ]
        if nano_legacy_out is not None:
            results.append(
                Result(
                    "Nano-vLLM (legacy)",
                    int(nano_legacy_out["output_tokens"]),
                    float(nano_legacy_out["time_s"]),
                )
            )
        _print_table(results)


if __name__ == "__main__":
    main()
