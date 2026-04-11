import torch 
from transformers import AutoTokenizer 
from dataclasses import dataclass


from vllm.engine.config import ModelConfig
from vllm.models.llama import LlamaForCausalLM
from vllm.loader import load_models
from vllm.engine.core.cache import KVCache, BlockKCache
from vllm.engine.core.block_manager import BlockManager
from vllm.engine.core.block import BLOCK_SIZE, BlockTable, compute_blocks
from vllm.sampler import Sampler
from vllm.engine.core.scheduler import Scheduler, SchedulerOutputs, SchedulingPolicy
from vllm.engine.core.sequence import  Sequence, SequenceStatus


@dataclass
class GenerationOutput:
    seq_id: int 
    prompt: str
    generated_text: str 
    prompt_tokens: int 
    generated_tokens: int 

class LLMEngine:
    """"Supports:
    - processing multiple requests simultienously (continous batching)
    - iteration-level scheduling. 
    - new requests can join mid-generation
    - completed  requests can leave immediately """


    def __init__(
            self,
            model_path,
            device = "cuda", 
            dtype = torch.float16,
            max_seq_len = 2048,
            max_batch_size = 8,
            use_paged_attention = True,
            num_blocks = None, 
            block_size = BLOCK_SIZE,
            scheduling_policy = SchedulingPolicy.PRIORITY,
            enable_preemption = True,
            max_prefill_tokens = 512, 
            enable_prefix_caching = True,
            use_flash_attn = True,):
        
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.use_paged_attention = use_paged_attention
        self.block_size = block_size
        self.scheduling_policy = scheduling_policy
        self.enable_preemption = enable_preemption
        self.max_prefill_tokens = max_prefill_tokens
        self.enable_prefix_caching = enable_prefix_caching
        self.use_flash_attn = use_flash_attn

        print(f"Loading mdoel from {model_path}")
        self.model = load_models(model_path, device=device, dtype=dtype)
        self.config = self.model.config
        print(f"Model loaded: {self.config.num_hidden_layers} layers, {self.config.hidden_size} hidden_size")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sampler = Sampler()

        if use_paged_attention:
            if num_blocks is None:
                #calculate the number of blocks
                num_blocks = self._calculate_num_blocks()
            print(f"Using pagedAttention with {num_blocks} blocks, block_size ={block_size}")

            self.block_manager = BlockManager(
                block_size=BLOCK_SIZE, 
                num_blocks=num_blocks,
                enable_prefix_caching=enable_prefix_caching
            )

            self.block_kv_cache = BlockKCache.from_config(
                config=self.config,
                num_blocks = num_blocks,
                block_size = block_size,
                device = device,
                dtype = dtype,
                )
            print(f"BlockKVCache memory: {self.block_kv_cache.memory_usage_mb:.1f}MB")

            self.scheduler = Scheduler(
                max_batch_size=max_batch_size,
                block_manager=self.block_manager,
                block_size=block_size,
                scheduling_policy=scheduling_policy,
                enable_preemption=enable_preemption,
                max_prefill_tokens=max_prefill_tokens,
            )

        else:
            # LEGACY KV CACHE MODE
            self.block_manager = None
            self.block_kv_cache = None 
            self.scheduler = Scheduler(
                max_batch_size=max_batch_size,
                scheduling_policy=scheduling_policy,
                enable_preemption=False,
                max_prefill_tokens = max_prefill_tokens
            )
        self._prompts = {}


    def _calculate_num_blocks(self):
        """Caluclate the number of blocks based on the available memory"""
        blocks_per_seq = (self.max_seq_len + self.block_size -1) // self.block_size
        return blocks_per_seq * self.max_batch_size
    

    def add_request(self,
                    prompt,
                    max_tokens = 100,
                    priority = 0):
        prompt_token_ids = self.tokenizer.encode(prompt)
        if len(prompt_token_ids) >= self.max_seq_len:
            raise ValueError(f"Prompt length {len(prompt_token_ids)} exceeds max_seq_len {self.max_seq_len}")
        
        seq = self.scheduler.add_request(prompt_token_ids, max_tokens, priority=priority)
        self._prompts[seq.seq_id] = prompt

        return seq.seq_id
    
    @torch.inference_mode()
    def step(self):
        