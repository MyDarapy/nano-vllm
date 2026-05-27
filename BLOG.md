### Building a Mini vLLM from Scratch: A deep dive into LLM Inference Optimization
# Nano-vLLM

A minimal, from-scratch LLM inference engine inspired by vLLM.

This project rebuilds the core ideas behind modern high-throughput inference systems in a compact and readable codebase. Instead of treating inference as a black box, Nano-vLLM makes the moving pieces visible: request scheduling, prefill vs decode execution, KV cache management, paged attention, prefix caching, and token sampling.

The goal is not just to generate text. The goal is to understand why fast LLM inference is hard, and how systems like vLLM solve it.

## Why This Exists

Running large language models efficiently is a systems problem, not just a modeling problem.

During generation, a model goes through two very different phases:

1. Prefill: process the full prompt in parallel
2. Decode: generate one token at a time while attending over everything seen so far

The decode phase is where naive implementations become expensive. If every request gets one large contiguous KV cache buffer sized for the maximum sequence length, memory usage balloons quickly and batching becomes harder. In practice, prompt lengths differ, requests finish at different times, and memory fragmentation starts to hurt throughput.

Nano-vLLM exists to show how an inference engine can do better:

- continuous batching instead of one-request-at-a-time execution
- paged KV cache allocation instead of one giant contiguous cache per request
- prefix caching so identical prompt prefixes can share work
- separate prefill and decode execution paths tuned for their very different workloads
- a simple scheduler that can admit, batch, preempt, and retire requests dynamically

It is a teaching project, but it is also a real engine with a clean execution model and a clear systems story.

## Key Features

- PagedAttention-style KV cache management using fixed-size blocks
- Continuous batching with iteration-level scheduling
- Separate prefill and decode paths
- Prefix caching for shared prompt prefixes
- Dynamic request admission and immediate ejection of finished sequences
- Optional priority-based scheduling with preemption
- Legacy per-sequence KV cache mode for debugging and comparison
- Llama-style model implementation from scratch
- Hugging Face config and safetensors loading
- Greedy decoding out of the box, with a sampler abstraction ready for extension

## What Nano-vLLM Implements

At a high level, the engine supports two execution backends.

### 1. Paged mode

This is the high-performance path:

- a global block KV cache is allocated up front
- each sequence owns a `BlockTable` that maps logical token positions to physical cache blocks
- prefill writes prompt KV states into block slots
- decode reads historical KV states through block tables rather than contiguous memory
- shared full blocks can be reused through prefix caching

### 2. Legacy mode

This is the simple reference path:

- each sequence owns its own contiguous `KVCache`
- attention uses the standard cached key/value tensors directly
- the mental model is much simpler, which makes it a good debugging baseline

Together, these two modes make it easier to compare a straightforward cache design with a paged, memory-efficient one.

## Architecture Overview

The project is organized around a small set of engine and memory-management primitives:

```text
vllm/
├── attention/
│   ├── flash_attention.py      # FlashAttention-style prefill kernel
│   └── paged_attention.py      # Paged decode attention kernel
├── core/
│   ├── block.py                # Block and BlockTable abstractions
│   ├── block_manager.py        # KV block allocation, freeing, prefix cache
│   ├── cache.py                # Legacy KVCache and BlockKCache
│   ├── kv_scatter.py           # Scatter new K/V states into block cache
│   ├── scheduler.py            # Continuous batching and preemption logic
│   └── sequence.py             # Per-request state tracking
├── layers/
│   ├── activations.py
│   ├── rmsnorm.py
│   └── rope.py
├── models/
│   ├── llama.py                # Llama model implementation
│   └── moe.py
├── config.py                   # Model and attention metadata
├── engine.py                   # Main inference engine
├── loader.py                   # HF config + safetensors loading
├── sampler.py                  # Token selection
├── batched.py                  # Batched-engine experimentation
└── utils.py                    # Padding helpers
```

The engine itself is intentionally small. Most of the interesting behavior comes from the interaction between just a few objects:

- `LLMEngine`: owns the tokenizer, model, scheduler, sampler, and cache backend
- `Scheduler`: decides which requests prefill, decode, wait, or get preempted each step
- `Sequence`: holds prompt tokens, generated tokens, priority, cache state, and progress
- `BlockManager`: allocates and frees physical KV blocks and tracks prefix-cache sharing
- `BlockKCache` / `KVCache`: the two cache backends
- `LlamaForCausalLM`: runs embeddings, attention, MLPs, normalization, and logits projection

## How Generation Works

The runtime loop follows the same basic pattern on every engine step:

1. Admit waiting requests into the active set, subject to batch and memory limits
2. Run prefill for new or chunked sequences
3. Run one decode step for active decoding sequences
4. Sample next tokens
5. Mark completed requests as finished and free their memory immediately

In paged mode, this produces a clean division of labor:

- prefill is compute-heavy and writes KV states into the block cache
- decode is memory-heavy and gathers historical KV states through block tables

That split is the core reason inference engines need dedicated systems design rather than a naive "just call the model again" loop.

## Core Ideas

### PagedAttention

Traditional KV caching assumes each request gets one contiguous memory allocation sized to the maximum length. That is simple, but wasteful.

Nano-vLLM instead divides the global KV cache into fixed-size blocks. A sequence does not need one giant contiguous memory region. It only needs a logical-to-physical mapping:

- logical block index = token position divided by block size
- physical block id = actual cache block where those KV states live
- slot mapping = exact position inside the global block cache where a new token should write

This gives the engine the flexibility to pack many requests into a shared memory pool without requiring contiguous space per request.

### Prefix Caching

If two prompts share the same full prefix blocks, there is no reason to recompute or duplicate those blocks. The `BlockManager` hashes full blocks and lets later requests reuse them by incrementing reference counts instead of allocating new memory.

That makes repeated prompts, shared system prompts, and common instruction prefixes much cheaper.

### Continuous Batching

Inference workloads are dynamic. Some requests are still prefilling while others are already decoding. Some finish early. New ones may arrive at any moment.

Instead of forming one static batch and waiting for everyone to finish, Nano-vLLM schedules work one iteration at a time:

- new requests can join while old ones are still running
- finished requests leave immediately
- the scheduler keeps decode work moving while admitting more prefill work when budget allows

### Priority Scheduling and Preemption

The scheduler can operate in either FCFS or priority mode. In priority mode, high-priority requests can displace lower-priority ones when memory pressure would otherwise block admission.

This is a small but useful demonstration of the fact that good inference engines are part model runtime and part resource manager.

### Two KV Cache Backends

The project intentionally keeps both cache implementations:

- `KVCache`: simple, contiguous, per-sequence, easy to reason about
- `BlockKCache`: shared, paged, memory-efficient, closer to a production inference engine

This dual-path design is useful both pedagogically and practically:

- legacy mode is the easiest place to debug correctness
- paged mode is where the real systems ideas show up

## Model Architecture

The repo includes a compact Llama-style causal LM implementation with the main architectural pieces exposed directly in Python:

- RMSNorm
- rotary position embeddings
- grouped-query attention
- SwiGLU-style MLP
- decoder-only transformer stack

The attention module is also where the runtime backend switches happen:

- in paged mode, K/V states are scattered into the global block cache and decode reads through paged attention
- in legacy mode, each sequence updates its own contiguous cache and attention runs against that cache directly

## Installation

Clone the repository and install the runtime dependencies:

```bash
git clone <your-repo-url>
cd vllm
pip install -r requirements.txt
```

The current dependency set is intentionally small:

- `torch`
- `transformers`
- `huggingface_hub`
- `safetensors`
- `triton`

For the paged-attention path, you will typically want:

- a CUDA-capable GPU
- a compatible PyTorch installation
- Triton available in your environment

If you just want to validate model logic or compare behavior, the legacy KV-cache path is the simplest starting point.

## Model Weights

Nano-vLLM accepts either:

- a Hugging Face model id
- a local directory containing `config.json` and `*.safetensors`

Because the loader uses `huggingface_hub.snapshot_download`, you can point the engine directly at a Hugging Face repo id:

```python
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

Or download weights locally first:

```bash
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir ./models/TinyLlama-1.1B-Chat-v1.0
```

Then pass the local path to the engine.

## Quick Start

### Single prompt generation

```python
import torch

from vllm.engine import LLMEngine
from vllm.core.scheduler import SchedulingPolicy

engine = LLMEngine(
    model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda",
    dtype=torch.float16,
    max_seq_len=2048,
    max_batch_size=8,
    use_paged_attention=True,
    scheduling_policy=SchedulingPolicy.PRIORITY,
    enable_preemption=True,
    enable_prefix_caching=True,
    use_flash_attn=True,
)

text = engine.generate(
    "The capital of France is",
    max_tokens=64,
)

print(text)
```

### Batched generation

```python
import torch

from vllm.engine import LLMEngine

engine = LLMEngine(
    model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda",
    dtype=torch.float16,
    max_batch_size=8,
    use_paged_attention=True,
)

prompts = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "Python is a programming language that",
]

outputs = engine.generate_batch(prompts, max_tokens=64)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print()
```

### Priority scheduling

If you want to see the scheduler more directly, add requests manually and drive the engine step-by-step:

```python
import torch

from vllm.engine import LLMEngine
from vllm.core.scheduler import SchedulingPolicy

engine = LLMEngine(
    model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda",
    dtype=torch.float16,
    use_paged_attention=True,
    scheduling_policy=SchedulingPolicy.PRIORITY,
    enable_preemption=True,
)

engine.add_request("Low priority request", max_tokens=32, priority=1)
engine.add_request("Urgent request", max_tokens=32, priority=10)

while engine.scheduler.has_pending_requests():
    finished = engine.step()
    for output in finished:
        print(output.seq_id, output.generated_text)
```

### Legacy KV-cache mode

If you want the simplest mental model, switch off paged attention:

```python
import torch

from vllm.engine import LLMEngine

engine = LLMEngine(
    model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cpu",
    dtype=torch.float32,
    use_paged_attention=False,
    use_flash_attn=False,
)

print(engine.generate("Hello from legacy mode", max_tokens=32))
```

This mode is especially useful when validating correctness, inspecting cache behavior per sequence, or bringing the system up without Triton kernels.

## Configuration Knobs

The most important engine options are:

- `max_seq_len`: maximum total sequence length the engine plans around
- `max_batch_size`: maximum number of active sequences in one scheduling step
- `use_paged_attention`: selects paged mode or legacy contiguous-cache mode
- `num_blocks`: total number of physical KV blocks in paged mode
- `block_size`: number of tokens stored per KV block
- `scheduling_policy`: `FCFS` or `PRIORITY`
- `enable_preemption`: whether high-priority requests can displace lower-priority ones
- `max_prefill_tokens`: budget used to decide how much prompt work to admit in a step
- `enable_prefix_caching`: whether shared prompt prefixes reuse full blocks
- `use_flash_attn`: whether to enable the flash-attention prefill path

These are enough to let you experiment with the main systems tradeoffs:

- memory efficiency vs simplicity
- prefill throughput vs decode responsiveness
- fairness vs priority
- cache sharing vs isolated execution

## Inspecting the Engine

You can ask the engine for a compact runtime summary:

```python
stats = engine.get_stats()
for key, value in stats.items():
    print(f"{key}: {value}")
```

That includes model, scheduler, and cache information such as:

- number of layers
- attention head counts
- max sequence length
- scheduling mode
- total and free KV blocks
- prefix cache statistics
- approximate KV cache memory footprint


## Performance Philosophy

Nano-vLLM is designed around the same high-level ideas that make vLLM fast:

- maximize useful batching
- avoid wasteful KV cache layouts
- separate compute-heavy prompt processing from memory-heavy decoding
- reuse work where prompt prefixes overlap

The code is intentionally small and educational, but the architecture is real. It is a useful foundation for experimenting with:

- higher-throughput batching strategies
- more advanced sampling
- better kernel implementations
- alternative schedulers
- serving-layer extensions

## References and Inspiration

This project is heavily inspired by the ideas popularized by:

- [vLLM](https://github.com/vllm-project/vllm)
- the PagedAttention paper and related engineering writeups
- FlashAttention-style efficient attention kernels
- the growing ecosystem of educational inference-engine rebuilds

