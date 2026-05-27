# Nano-vLLM

A lightweight vLLM-inspired inference engine built from scratch. 

## Key Features
 
- 🧠 **Paged KV cache**: memory efficient KV cache management
- 🚀 **Continuous batching**: process multiple requests simulatenously with iteration-level scheduling
- 🧩 **Prefix caching**: reuse shared full-prefix blocks across requests
- ⚡ **FlashAttention Triton kernels**: fused attention kernels for paged prefill and paged decode 
- 🧪 **Legacy mode**: contiguous per-sequence KV cache for debugging and comparison

## Installation

```bash
git clone <https://github.com/MyDarapy/nano-vllm/>
cd nano-vllm
pip install -r requirements.txt
```


## Quick Start

```python
import torch
from vllm.engine import LLMEngine

engine = LLMEngine(
    model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda",
    dtype=torch.float16,
    max_seq_len=2048,
    max_batch_size=8,
    use_paged_attention=True,
)

print(engine.generate("Hello from Nano-vLLM!", max_tokens=64))
```

## Benchmark (Nano-vLLM vs official vLLM)

This repo’s package name is `vllm/`, which conflicts with the official PyPI `vllm`. The benchmark runs each engine in its own venv via subprocess:

- `benchmarks/compare_official_vllm.py`

Small sanity benchmark:

```bash
python3 benchmarks/compare_official_vllm.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --nano-python /root/venvs/nano/bin/python \
  --vllm-python /root/venvs/vllm/bin/python \
  --num-seqs 5 --min-input-len 10 --max-input-len 30 \
  --min-output-len 5 --max-output-len 20 --max-model-len 128 \
  --max-batch-size 5 --max-prefill-tokens 256
```
Benchmark Configurations
- Hardware: RTX 4000 Ada (20GB)
- Model: TinyLlama-1.1B-Chat-v1.0

## Performance Results

<table>
  <thead>
    <tr>
      <th>Inference Engine</th>
      <th align="right">Output Tokens</th>
      <th align="right">Time (s)</th>
      <th align="right">Throughput (tokens/s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>vLLM</td>
      <td align="right">73</td>
      <td align="right">0.17</td>
      <td align="right">440.45</td>
    </tr>
    <tr>
      <td>Nano-vLLM (paged)</td>
      <td align="right">73</td>
      <td align="right">0.39</td>
      <td align="right">188.80</td>
    </tr>
    <tr>
      <td>Nano-vLLM (legacy)</td>
      <td align="right">73</td>
      <td align="right">1.21</td>
      <td align="right">60.08</td>
    </tr>
  </tbody>
</table>


## Docs

See `BLOG.md` for a deeper architecture walkthrough.

## Want to contribute? 
Contributions are welcome. Nano‑vLLM started as an educational project, but the focus is shifting toward performance optimization while keeping the codebase readable. Here are high impact areas where help is valuable: 

- Write/optimize custom kernels (Triton/CUDA) for hot-path ops to push throughput and latency closer to official vLLM implementation. Current implementation only has kernel support for attention operations. Ops like MLP, RMSNorm, etc needs kernel their own custom kernels.
- Add support for more architectures beyond Llama-style models (e.g., Qwen, Kimi, etc.).
- Fix chunked prefill so attention can correctly attend to the full prefix (not just the current chunk).
- Implement support for topk sampling and nucleus sampling


