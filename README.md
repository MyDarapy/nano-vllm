

### TODO
- implement model loader from hf and local paths 
- Add more sampling strategy (topk, top-p). Current implementation only surports greedy decoding 
- implement spectulative decoding 
- CUDA graphs optim to launch triton kernels

Supports
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
