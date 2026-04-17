import torch 
from transformers import AutoTokenizer 
from dataclasses import dataclass


from vllm.engine.config import ModelConfig, Metadata
from vllm.models.llama import LlamaForCausalLM
from vllm.loader import load_models
from vllm.engine.core.cache import KVCache, BlockKCache
from vllm.engine.core.block_manager import BlockManager
from vllm.engine.core.block import BLOCK_SIZE, BlockTable, compute_blocks
from vllm.sampler import Sampler
from vllm.engine.core.scheduler import Scheduler, SchedulerOutputs, SchedulingPolicy
from vllm.engine.core.sequence import  Sequence, SequenceStatus
from vllm.utils import input_padding

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
        scheduler_outputs = self.scheduler.schedule()
        if scheduler_outputs.is_empty():
            return []
        
        completed_outputs = []
        # TODO: this can definitely be paralized with batching to take advantage of the GPUs compute unit 
        for i, seq in enumerate(scheduler_outputs.chunked_prefill_sequences):
            num_tokens = scheduler_outputs.chunked_prefill_tokens[i]
            self._run_chunked_prefill_paged(seq, num_tokens)

        # TODO: parallelize prefill across a batch. currently highly inefficient with the for loop
        for seq in scheduler_outputs.prefill_sequences:
            self._run_prefill(seq) # (runs batch_size amount of forward pass(inefficient, parallelize later))

    def _run_prefill(self, seq):
        if self.use_paged_attention:
            self._run_prefill_paged(seq)
        else:
            self._run_prefill_legacy(seq)

    def _run_prefill_legacy(self, seq):
        pass

    def _run_prefill_paged(self, sequence): 
        """Per sequence prefill for simplicity but compute units sip juice"""
        prompt_len = sequence.get_prompt(len)
        num_blocks_needed = compute_blocks(prompt_len, self.block_size)
        sequence.block_table, sequence.shared_prefix_len = (
            self.block_manager.allocate_block_with_prefix_caching(sequence.prompt_token_ids))
        
        if sequence.shared_prefix_len > 0 and sequence.shared_prefix_len < prompt_len:
            # Partial cache hit process only the remaining non-cache token
            token_to_process = sequence.prompt_token_ids[sequence.shared_prefix_len:]
            start_position = sequence.shared_prefix_len
        elif sequence.shared_prefix_len >= prompt_len
            # Full cache hit we only need to run forward for for the last token 
            # KV cache is already populated for all prompt tokens 
            tokens_to_process = [sequence.prompt_token_ids[-1]]
            start_position = prompt_len - 1
        else:
            # No cache hit
            tokens_to_process = sequence.prompt_token_ids 
            start_position = 0 
        
        input_ids = torch.tensor([tokens_to_process], dtype=torch.long, device=self.device)
        context_len = [prompt_len]
        slot_mapping = sequence.block_table.slot_mapping(prompt_len)

        metadata = Metadata(
            is_prefill=True,
            block_tables=[sequence.block_table],
            context_len=context_len,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
            positions=start_position
        )
        #Forward pass for PagedAttention prefill
        logits = self.model(input_ids, metadata, kv_cache=self.block_kv_cache)
        
        # Sample first (next) token 
        next_token = self.sampler.greedy_decoding(logits)
        sequence.append_token(next_token.item())

    def _run_chunked_prefill_paged(self, sequence, num_tokens):
        start_pos = sequence.num_prefilled_tokens
        end_pos = start_pos + num_tokens
        chunk_tokens = sequence.prompt_token_ids[start_pos:end_pos]

        # calculate the amount of blocks needed for this chunk
        total_tokens_after = end_pos 
        blocks_needed = compute_blocks(total_tokens_after, sequence.block_size)

        if sequence.block_table is None:
            sequence.block_table = self.block_manager.allocate_blocks_for_sequence(blocks_needed)
        else:
            current_blocks = sequence.block_table.num_blocks()
            new_block_needed = blocks_needed - current_blocks
            for _ in range (new_block_needed):
                block_id = self.block_manager.allocate_block()
                sequence.block_table.append_block(block_id)

        input_ids = torch.tensor([chunk_tokens], dtype=torch.long, device=self.device)
        context_lens = [end_pos] #total token we will have after this chunk 
        start_position = [start_pos]
        slot_mapping = sequence.block_table.slot_mapping_range(start_pos, end_pos)
        positions = torch.arange(start_pos, end_pos, device=self.device).unsqueeze(1)

        metadata = Metadata(
            is_prefill=True,
            block_table = [sequence.block_table],
            context_lens=context_lens,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
            positions=positions
        )
        logits = self.model(input_ids, metadata, kv_cache=self.block_kv_cache)

        # Update prefill progress
        sequence.num_prefill_tokens = end_pos

        #If all prompt tokens are processed sample the first output token
        if sequence.num_prefilled_tokens >= len(sequence.prompt_token_ids):
            next_token = self.sampler.greedy_decoding(logits)
            sequence.append_token(next_token.item())


    def _run_batched_chuncked_prefill():
        pass

    def run_decode(self, sequences):
        """Run decode on batched sequences one token per sequence"""
        if self.use_paged_attention:
            self._run_batched_decode_paged(sequences)
        else:
            self._run_decode_legacy

    def run_batched_decode_paged(self, sequences):
        batch_size = len(sequences)
        for seq in sequences:
            new_blocks_needed = seq.get_num_new_blocks_needed(self.block_size)
            if new_blocks_needed > 0:
                for _ in range(new_blocks_needed):
                    block_id = self.block_manager.allocate_block()
                    seq.block_table.append(block_id)

        #prepare batched input = last token id from each sequence
        input_ids = torch.tensor([seq.get_last_token_id() for seq in sequences], 
                                 dtype= torch.long,
                                 device=self.device,
                                ) # [B, 1]
        block_tables 
        slot_mapping = 

    def run_decode_legacy(self, sequences):
        pass

    def _create_output(self, seq):
        pass

    def _run_to_completion(self):
        pass

    def generate(self, prompt, max_tokens=100):
        pass

    def generate_batch(self, prompts, max_tokens=100):
        pass

    def get_stats(self):
        pass

    def _run_batched_prefill_paged(self, sequences):
        """"Batched prefill (offers better throughput and GPU compute unit usage)"""
        """"Make GPU inference go brrrrrr"""
        token_batches = []
        seq_lens_list = []

        # allocate blocks upfront and collect prompt tokens 
        for seq in sequences:
            prompt_len = seq.get_prompt_len()
            num_blocks_needed = compute_blocks(prompt_len, self.block_size)

            if seq.block_table is None:
                seq.block_table = self.block_manager.allocate_blocks_for_sequence(num_blocks_needed)
            token_batches.append(seq.prompt_token_ids)
            seq_lens_list.append(prompt_len)

            input_ids, seq_lens = input_padding(token_batches, 
                                             pad_value=self.tokenizer.pad_token_id)

            slot_mapping = seq.block_table

            metadata = Metadata(
                is_prefill= True,
                block_tables=None,)
        pass




