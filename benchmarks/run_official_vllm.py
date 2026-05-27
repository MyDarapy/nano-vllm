from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload-json", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true", default=False)
    ap.add_argument("--warmup-steps", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--ignore-eos", action="store_true", default=True)
    ap.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    ap.add_argument(
        "--disable-chunked-prefill",
        action="store_true",
        default=True,
        help="Best-effort: disable chunked prefill (works reliably under V0; V1 may ignore).",
    )
    ap.add_argument("--enable-chunked-prefill", dest="disable_chunked_prefill", action="store_false")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # type: ignore
    import inspect

    with open(args.workload_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    prompt_token_ids = payload["prompt_token_ids"]
    max_tokens_list = payload["max_tokens_list"]

    llm_kwargs = {
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
    }
    sig = inspect.signature(LLM.__init__)
    if "enable_chunked_prefill" in sig.parameters:
        llm_kwargs["enable_chunked_prefill"] = (not args.disable_chunked_prefill)
    llm = LLM(args.model, **llm_kwargs)

    for _ in range(args.warmup_steps):
        llm.generate(["Benchmark: "], SamplingParams(max_tokens=1))

    sampling_params = [
        SamplingParams(temperature=args.temperature, ignore_eos=args.ignore_eos, max_tokens=m)
        for m in max_tokens_list
    ]

    t0 = time.time()
    try:
        llm.generate(
            prompts=None,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            use_tqdm=False,
        )
    except TypeError:
        # Older APIs accept list[dict] inputs.
        inputs = [{"prompt_token_ids": p} for p in prompt_token_ids]
        llm.generate(inputs, sampling_params, use_tqdm=False)
    t = time.time() - t0

    total_requested = int(sum(max_tokens_list))
    print(json.dumps({"time_s": float(t), "output_tokens": total_requested}))


if __name__ == "__main__":
    main()
