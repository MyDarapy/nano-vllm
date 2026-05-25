from vllm.engine import LLMEngine
from vllm.core.scheduler import SchedulingPolicy
import torch 

DEVICE="cuda"
DTYPE = torch.float16

engine = LLMEngine(
    model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    device=DEVICE,
    dtype = torch.float16,
    max_seq_len=2048,
    max_batch_size=8,
    use_paged_attention=True,
    use_flash_attn=True,
    scheduling_policy=SchedulingPolicy.PRIORITY,
)

requests = [
     ("Explain continuous batching in simple terms.", 64, 1),
    ("What is paged attention?", 64, 5),
    ("Write a short poem about Lagos rain.", 48, 2),
]

for prompt, max_tokens, priority in requests:
    seq_id= engine.add_request(
        prompt=prompt,
        max_tokens=max_tokens,
        priority=priority,
    )
    print(f"queued-seq_id={seq_id}")


while engine.scheduler.has_pending_requests():
    finished = engine.step()
    for out in finished:
        print("=" * 80)
        print(f"seq_id: {out.seq_id}")
        print(f"prompt: {out.prompt}")
        print(f"generated_text: {out.generated_text}")
        print(f"prompt_tokens: {out.prompt_tokens}")
        print(f"generated_tokens: {out.generated_tokens}")