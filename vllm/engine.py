import torch 
from transformers import AutoTokenizer 
from dataclasses import dataclass


from vllm.config import ModelConfig, Metadata
from vllm.models.llama import LlamaForCausalLM
from vllm.loader import load_models
from vllm.core.cache import KVCache, BlockKCache
from vllm.core.block_manager import BlockManager
from vllm.core.block import BLOCK_SIZE, BlockTable, compute_blocks
from vllm.sampler import Sampler
from vllm.core.scheduler import Scheduler, SchedulerOutputs, SchedulingPolicy
from vllm.core.sequence import  Sequence, SequenceStatus
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
        self.model = load_models(
            model_path,
            device=device,
            dtype=dtype,
            use_flash_attn=use_flash_attn,
        )
        self.config = self.model.config
        print(f"Model loaded: {self.config.num_hidden_layers} layers, {self.config.hidden_size} hidden_size")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"eos_token: {self.tokenizer.eos_token}")
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
                max_prefill_tokens=max_seq_len,
            )
        self._prompts = {}


    def _calculate_num_blocks(self):
        """Caluclate the number of blocks based on the available memory"""
        blocks_per_seq = (self.max_seq_len + self.block_size -1) // self.block_size
        return blocks_per_seq * self.max_batch_size

    def _build_positions(self, start: int, length: int) -> torch.Tensor:
        return torch.arange(
            start,
            start + length,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
    

    def add_request(self,
                    prompt,
                    max_tokens = 100,
                    priority = 0):
        messages = [{"role": "user", "content":prompt}]
        prompt_token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        print(prompt_token_ids)
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
        # process chunked prefill sequences 
        # TODO: this can definitely be paralized with batching to take advantage of the GPUs compute unit 
        for i, seq in enumerate(scheduler_outputs.chunked_prefill_sequences):
            num_tokens = scheduler_outputs.chunked_prefill_tokens[i]
            self._run_chunked_prefill_paged(seq, num_tokens)
        
        # process full sequence one at a time for simplicity
        # TODO: parallelize prefill across a batch. currently highly inefficient with the for loop
        for seq in scheduler_outputs.prefill_sequences:
            self._run_prefill(seq) # (runs batch_size amount of forward pass(inefficient, parallelize later))

        # proces batched decode sequences
        if scheduler_outputs.decode_sequences:
            self.run_decode(scheduler_outputs.decode_sequences)
        
        # Chck for finished sequence 
        newly_finished = self.scheduler.update_sequences(self.tokenizer.eos_token_id)

        """if not newly_finished:
            return """ 
        for seq in newly_finished:
            output = self._create_output(seq)
            completed_outputs.append(output) # Each item in completed_output is a GenerationOutput object

            #Free blocks for new sequences
            if self.use_paged_attention and seq.block_table is not None:
                self.block_manager.free_sequence_blocks(seq.block_table)
                seq.block_table = None
        
        return completed_outputs

    def _run_prefill(self, seq):
        if self.use_paged_attention:
            self._run_prefill_paged(seq)
        else:
            self._run_prefill_legacy(seq)

    def _run_prefill_paged(self, sequence): 
        """Per sequence prefill for simplicity but compute units sip juice"""
        prompt_len = sequence.get_prompt_len()
        sequence.block_table, sequence.shared_prefix_len = (
            self.block_manager.allocate_block_with_prefix_caching(sequence.prompt_token_ids))
        
        if sequence.shared_prefix_len > 0 and sequence.shared_prefix_len < prompt_len:
            # Partial cache hit process only the remaining non-cache token
            token_to_process = sequence.prompt_token_ids[sequence.shared_prefix_len:]
            start_position = sequence.shared_prefix_len
        elif sequence.shared_prefix_len >=prompt_len:
            # Full cache hit we only need to run forward for for the last token 
            # KV cache is already populated for all prompt tokens 
            tokens_to_process = [sequence.prompt_token_ids[-1]]
            start_position = prompt_len - 1
        else:
            # No cache hit
            tokens_to_process = sequence.prompt_token_ids 
            start_position = 0 
        
        input_ids = torch.tensor([tokens_to_process], dtype=torch.long, device=self.device)
        slot_mapping = sequence.block_table.slot_mapping_range(
            start_position,
            start_position + len(tokens_to_process),
        )

        metadata = Metadata(
            is_prefill=True,
            block_tables=None,
            context_lens=torch.tensor([prompt_len], dtype=torch.long, device=self.device),
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
            positions=self._build_positions(start_position, len(tokens_to_process)),
        )
        #Forward pass for PagedAttention prefill
        logits = self.model(input_ids, metadata, kv_cache=self.block_kv_cache)
        
        # Sample first (next) token 
        next_token = self.sampler.greedy_decoding(logits)
        sequence.append_token(next_token.item())


    def _run_batched_paged_prefill(self, sequences):
        if not sequences:
            return 
        B = len(sequences)
        prompt_lens = [seq.get_prpmpt_len() for seq in sequences]
        T = max(prompt_lens)

        for seq, seq_len in zip(sequences, prompt_lens):
            if seq.block_table is None:
                num_blocks_needed = compute_blocks(seq_len, self.block_size)
                seq.block_table = self.block_manager.allocate_blocks_for_sequence(num_blocks_needed)

        input_ids = torch.full((B, T), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        positions = torch.zeros((B, T), dtype=torch.long, device=self.device)
        slot_mapping = torch.full((B, T), -1, dtype=torch.int32, device=self.device)

        for b, (seq, seq_len) in enumerate(zip(sequences, prompt_lens)):
            tokens = seq.prompt_token_ids 
            input_ids[b, :seq_len] = torch.tensor(tokens, dtype= torch.long, device=self.device)
            positions[b, :seq_len] = torch.arange(0, seq_len, dtype=torch.long, device=self.device)

            slots = seq.block_table.slot_mapping_range(0, seq_len)
            slot_mapping[b, :seq_len] = torch.tensor(slots, dtype=torch.long, device=self.device)

        context_lens = torch.tensor(prompt_lens, dtype=torch.long, device=self.device)

        metadata = Metadata(
            is_prefill=True,
            block_tables=None,
            context_lens=context_lens,
            slot_mapping=slot_mapping,   # will be flattened inside store_kvcache via reshape(-1)
            positions=positions,)
        
        logits = self.model(input_ids, metadata, kv_cache=self.block_kv_cache)
        next_tokens = []
        for b, seq_len in enumerate(prompt_lens):
            last_logits = logits[b, seq_len-1, :]
            next_tokens.append(int(torch.argmax(last_logits).item()))
        
        for seq, tok in zip(sequences, next_tokens):
            seq.append_token(tok)

    def _run_chunked_prefill_paged(self, sequence, num_tokens):
        start_pos = sequence.num_prefilled_tokens
        end_pos = start_pos + num_tokens
        chunk_tokens = sequence.prompt_token_ids[start_pos:end_pos]

        # calculate the amount of blocks needed for this chunk
        total_tokens_after = end_pos 
        blocks_needed = compute_blocks(total_tokens_after, self.block_size)

        if sequence.block_table is None:
            sequence.block_table = self.block_manager.allocate_blocks_for_sequence(blocks_needed)
        else:
            current_blocks = sequence.block_table.num_blocks()
            new_block_needed = blocks_needed - current_blocks
            for _ in range (new_block_needed):
                block_id = self.block_manager.allocate_block()
                sequence.block_table.append_block(block_id)

        input_ids = torch.tensor([chunk_tokens], dtype=torch.long, device=self.device)
        context_lens = torch.tensor([end_pos], dtype=torch.long, device=self.device)
        slot_mapping = sequence.block_table.slot_mapping_range(start_pos, end_pos)
        positions = self._build_positions(start_pos, len(chunk_tokens))

        metadata = Metadata(
            is_prefill=True,
            block_tables=None,
            context_lens=context_lens,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
            positions=positions
        )
        logits = self.model(input_ids, metadata, kv_cache=self.block_kv_cache)

        # Update prefill progress
        sequence.num_prefilled_tokens = end_pos

        #If all prompt tokens are processed sample the first output token
        if sequence.num_prefilled_tokens >= len(sequence.prompt_token_ids):
            next_token = self.sampler.greedy_decoding(logits)
            sequence.append_token(next_token.item())

    def run_decode(self, sequences):
        """Run decode on batched sequences one token per sequence"""
        if self.use_paged_attention:
            self.run_batched_decode_paged(sequences)
        else:
            self.run_decode_legacy(sequences)

    def run_batched_decode_paged(self, sequences):
        batch_size = len(sequences)
        for seq in sequences:
            new_blocks_needed = seq.get_num_new_blocks_needed(self.block_size)
            if new_blocks_needed > 0:
                for _ in range(new_blocks_needed):
                    block_id = self.block_manager.allocate_block()
                    seq.block_table.append_block(block_id)

        #prepare batched input = last token id from each sequence
        input_ids = torch.tensor([[seq.get_last_token_id()] for seq in sequences], 
                                 dtype= torch.long,
                                 device=self.device,
                                ) # [B, 1]
        block_tables = [sequence.block_table for sequence in sequences]

        # absolute position of current decode token
        start_positions = [[sequence.get_len() - 1] for sequence in sequences] # [B, 1]
        
        context_lens = [sequence.get_len() for sequence in sequences] # [B]

        slot_mapping = []
        for sequence in sequences:
            # get the current position of this token in the sequence
            i_current_position = sequence.get_len() - 1
            slot_for_current_i = sequence.block_table.slot_mapping_for_pos(i_current_position)
            slot_mapping.append(slot_for_current_i)

        max_blocks = max(table.num_blocks() for table in block_tables)
        block_table_tensor = torch.full(
            (batch_size, max_blocks),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        for i, table in enumerate(block_tables):
            block_ids = torch.tensor(table.block_ids, dtype=torch.long, device=self.device)
            block_table_tensor[i, : table.num_blocks()] = block_ids

        metadata = Metadata(
            is_prefill=False,
            block_tables=block_table_tensor,
            context_lens=torch.tensor(context_lens, dtype=torch.long, device=self.device),
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
            positions=torch.tensor(start_positions, dtype=torch.long, device=self.device),
        )
        
        logits = self.model(
            input_ids, metadata, self.block_kv_cache
        )
        next_token = self.sampler.greedy_decoding(logits) #[batch_size]

        for i, seq in enumerate(sequences):
            seq.append_token(next_token[i].item())

    def _create_output(self, seq):
        """Output for a finished sequence"""
        all_tokens = seq.get_token_ids()
        generated_text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)

        return GenerationOutput(
            seq_id=seq.seq_id,
            prompt=self._prompts.get(seq.seq_id, ""),
            generated_text=generated_text,
            prompt_tokens= seq.get_prompt_len(),
            generated_tokens=seq.get_output_len(),
        )
    def _run_to_completion(self):
        """Run until all pending requests are complete"""
        all_outputs = []
        while self.scheduler.has_pending_requests():
            completed_outputs = self.step()
            all_outputs.extend(completed_outputs)

        all_outputs.sort(key=lambda x: x.seq_id)
        return all_outputs
        
    def generate(self, prompt, max_tokens=100):
        self.add_request(prompt, max_tokens)
        outputs = self._run_to_completion()
        return outputs[0].generated_text if outputs else ""

    def generate_batch(self, prompts, max_tokens=100):
        for prompt in prompts:
            self.add_request(prompt, max_tokens)
        
        outputs = self._run_to_completion()

        return [output.generated_text for output in outputs]

    def get_stats(self) -> dict:
        """Return engine statistics."""
        stats = {
            "model_layers": self.config.num_hidden_layers,
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "num_attention_heads": self.config.num_attention_heads,
            "num_kv_heads": self.config.num_kv_heads,
            "max_seq_len": self.max_seq_len,
            "max_batch_size": self.max_batch_size,
            "scheduler": str(self.scheduler),
            "use_paged_attention": self.use_paged_attention,
            # Advanced scheduling (Phase 4)
            "scheduling_policy": self.scheduling_policy.value,
            "enable_preemption": self.enable_preemption,
            "max_prefill_tokens": self.max_prefill_tokens,
            "enable_prefix_caching": self.enable_prefix_caching,
            # Optimizations (Phase 5)
            "use_flash_attn": self.use_flash_attn,
            "flash_attn_available": self.model.use_flash_attn if hasattr(self.model, 'use_flash_attn') else False,
        }

        # Add PagedAttention-specific stats
        if self.use_paged_attention:
            stats.update({
                "block_size": self.block_size,
                "total_blocks": self.block_manager.num_blocks,
                "free_blocks": self.block_manager.get_num_free_blocks(),
                "used_blocks": self.block_manager.num_blocks - self.block_manager.get_num_free_blocks(),
                "kv_cache_memory_mb": self.block_kv_cache.memory_usage_mb,
            })

            # Add prefix cache stats
            if self.enable_prefix_caching:
                prefix_stats = self.block_manager.get_prefix_cache_stat()
                stats.update({
                    "prefix_cache_blocks": prefix_stats["number_of_cached_blocks"],
                    "prefix_cache_refs": prefix_stats["total_reference"],
                })

        return stats

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

    def _run_prefill_legacy(self, seq):
        if seq.kv_cache is None:
            seq.kv_cache = KVCache(
                self.config,
                max_seq_len=self.max_seq_len,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            seq.kv_cache.reset()

        input_ids = torch.tensor([seq.prompt_token_ids], dtype=torch.long, device=self.device)
        metadata = Metadata(
            is_prefill=True,
            positions=self._build_positions(0, len(seq.prompt_token_ids)),
            context_lens=torch.tensor([len(seq.prompt_token_ids)], dtype=torch.long, device=self.device),
        )
        logits = self.model(input_ids, metadata, kv_cache=seq.kv_cache)
        next_token = self.sampler.greedy_decoding(logits)
        print(f"next_token: {next_token.item()}")
        seq.num_prefilled_tokens = len(seq.prompt_token_ids)
        seq.append_token(next_token.item())

    def run_decode_legacy(self, sequences):
        for seq in sequences:
            if seq.kv_cache is None:
                raise RuntimeError(f"Sequence {seq.seq_id} is missing a legacy KV cache")

            input_ids = torch.tensor([[seq.get_last_token_id()]], dtype=torch.long, device=self.device)
            metadata = Metadata(
                is_prefill=False,
                positions=torch.tensor([[seq.get_len() - 1]], dtype=torch.long, device=self.device),
                context_lens=torch.tensor([seq.get_len()], dtype=torch.long, device=self.device),
            )
            logits = self.model(input_ids, metadata, kv_cache=seq.kv_cache)
            next_token = self.sampler.greedy_decoding(logits)
            seq.append_token(next_token.item())

    def _run_batched_chuncked_prefill():
        pass
