

### TODO
- implement model loader from hf and local paths 
- Add more sampling strategy (topk, top-p nucleus sampling). Current implementation only surports greedy decoding 
- implement spectulative decoding 
- CUDA graphs optim to launch triton kernels
- Implement SPDA/FA with KV for legacy mode [CRITICAL]

Currently supports:
- Continous Batching
- Prefix caching 
- Immediate ejection of completed requests within a batch no waiting for the rest of the sequences in the batch to finish geberation. Completed requests leave immediately. 
- Dynamic batching
- Priority scheduling by preemepting low prority requests. Also supports First in First Out (FIFS) scheduling   
- Flash attention for the prefill stage
- PagedAttention 
- Also supports per sequence KVCache (Legacy (wasteful) KV caching)



### Checks 
- slot mapping concatenation from all the block tables from the different requests 


current implementation supports per sequence prefill for simplicity. Batch prefill and batched chunked prefill seems to take quite some brainpower to reason through. This will definetly affect throughput in the short term but in the spirit of moving fast and breaking things i will implement per sequence prefill and work on batched prefill later but decoding is batched. much easier to do
- per sequence prefill and batch prefill
