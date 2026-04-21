from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer

from nano_vllm.model.loader import load_model
from nano_vllm.sampler import Sampler
from nano_vllm.core.sequence import Sequence, SequenceStatus
from nano_vllm.core.scheduler import Scheduler, SchedulingPolicy
from nano_vllm.core.block import BLOCK_SIZE, compute_num_blocks
from nano_vllm.core.block_manager import BlockManager
from nano_vllm.cache import BlockKVCache


# -----------------------------
# Metadata contract
# -----------------------------

@dataclass
class AttentionMetadata:
    """
    Hybrid metadata:
    - sequence-level tensors for attention reads
    - token-level slot_mapping for KV scatter writes

    Shapes:
      block_tables:  [B, max_blocks] or None
      context_lens:  [B] or None
      seq_lens:      [B] or None
      slot_mapping:  [N] or None

    Where:
      B = number of sequences in current model forward
      N = total number of newly-written tokens in this forward
    """
    is_prefill: bool
    block_tables: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
    slot_mapping: Optional[torch.Tensor] = None


@dataclass
class GenerationOutput:
    seq_id: int
    prompt: str
    generated_text: str
    prompt_tokens: int
    generated_tokens: int


# -----------------------------
# Utility helpers
# -----------------------------

def pad_2d_int(seqs: List[List[int]], pad_value: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      padded_ids: [B, T]
      seq_lens:   [B]
    """
    if not seqs:
        return (
            torch.empty((0, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    lengths = [len(x) for x in seqs]
    max_len = max(lengths)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long, device=device)

    for i, s in enumerate(seqs):
        if len(s) > 0:
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

    return out, torch.tensor(lengths, dtype=torch.long, device=device)


def pad_block_tables(block_tables: List[List[int]], device: str) -> torch.Tensor:
    """
    block_tables -> [B, max_blocks], padded with -1
    """
    if not block_tables:
        return torch.empty((0, 0), dtype=torch.int32, device=device)

    max_len = max(len(bt) for bt in block_tables)
    out = torch.full((len(block_tables), max_len), -1, dtype=torch.int32, device=device)
    for i, bt in enumerate(block_tables):
        if len(bt) > 0:
            out[i, :len(bt)] = torch.tensor(bt, dtype=torch.int32, device=device)
    return out


def build_positions_from_lengths(lengths: torch.Tensor, device: str) -> torch.Tensor:
    """
    lengths: [B]
    returns positions: [B, T_max]
    position values are 0..len_i-1 for each sequence, padded region arbitrary (0)
    """
    if lengths.numel() == 0:
        return torch.empty((0, 0), dtype=torch.long, device=device)

    B = lengths.shape[0]
    T = int(lengths.max().item())
    pos = torch.zeros((B, T), dtype=torch.long, device=device)
    for i in range(B):
        L = int(lengths[i].item())
        if L > 0:
            pos[i, :L] = torch.arange(L, dtype=torch.long, device=device)
    return pos


def flatten_valid_kv(k: torch.Tensor, v: torch.Tensor, seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input:
      k: [B, T, Hkv, D]
      v: [B, T, Hkv, D]
      seq_lens: [B]

    Output:
      k_flat: [N, Hkv, D]
      v_flat: [N, Hkv, D]

    Flatten order:
      batch-major, then time-major within each sequence
    """
    chunks_k = []
    chunks_v = []
    B = k.shape[0]
    for i in range(B):
        L = int(seq_lens[i].item())
        if L > 0:
            chunks_k.append(k[i, :L])   # [L, Hkv, D]
            chunks_v.append(v[i, :L])
    if not chunks_k:
        Hkv = k.shape[2]
        D = k.shape[3]
        empty = torch.empty((0, Hkv, D), dtype=k.dtype, device=k.device)
        return empty, empty
    return torch.cat(chunks_k, dim=0), torch.cat(chunks_v, dim=0)


# -----------------------------
# LLM Engine
# -----------------------------

class LLMEngine:
    """
    Engine design:
      - batch full prefill
      - batch chunked prefill
      - batch decode
      - flatten only for KV scatter (slot_mapping)

    The model is assumed to accept:
      model(input_ids, metadata=..., kv_cache=...)

    and internally:
      - prefill uses FlashAttention
      - decode uses paged attention
      - store_kvcache uses metadata.slot_mapping
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_seq_len: int = 2048,
        max_batch_size: int = 8,
        use_paged_attention: bool = True,
        num_blocks: Optional[int] = None,
        block_size: int = BLOCK_SIZE,
        scheduling_policy: SchedulingPolicy = SchedulingPolicy.PRIORITY,
        enable_preemption: bool = True,
        max_prefill_tokens: int = 512,
        use_flash_attn: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.use_paged_attention = use_paged_attention
        self.block_size = block_size
        self.scheduling_policy = scheduling_policy
        self.enable_preemption = enable_preemption
        self.max_prefill_tokens = max_prefill_tokens
        self.use_flash_attn = use_flash_attn

        self.model = load_model(model_path, device=device, dtype=dtype, use_flash_attn=use_flash_attn)
        self.config = self.model.config

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sampler = Sampler()

        if num_blocks is None:
            num_blocks = self._calculate_num_blocks()

        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
        )

        self.block_kv_cache = BlockKVCache.from_config(
            config=self.config,
            num_blocks=num_blocks,
            block_size=block_size,
            device=device,
            dtype=dtype,
        )

        self.scheduler = Scheduler(
            max_batch_size=max_batch_size,
            block_manager=self.block_manager,
            block_size=block_size,
            scheduling_policy=scheduling_policy,
            enable_preemption=enable_preemption,
            max_prefill_tokens=max_prefill_tokens,
        )

        self._prompts: Dict[int, str] = {}

    def _calculate_num_blocks(self) -> int:
        blocks_per_seq = (self.max_seq_len + self.block_size - 1) // self.block_size
        return blocks_per_seq * self.max_batch_size

    def add_request(self, prompt: str, max_tokens: int = 100, priority: int = 0) -> int:
        prompt_token_ids = self.tokenizer.encode(prompt)

        if len(prompt_token_ids) >= self.max_seq_len:
            raise ValueError(
                f"Prompt length {len(prompt_token_ids)} exceeds max_seq_len={self.max_seq_len}"
            )

        seq = self.scheduler.add_request(prompt_token_ids, max_tokens, priority=priority)
        self._prompts[seq.seq_id] = prompt
        return seq.seq_id

    @torch.inference_mode()
    def step(self) -> List[GenerationOutput]:
        scheduler_outputs = self.scheduler.schedule()
        if scheduler_outputs.is_empty():
            return []

        completed_outputs: List[GenerationOutput] = []

        # 1) batched chunked prefill
        if scheduler_outputs.chunked_prefill_sequences:
            self._run_batched_chunked_prefill(
                scheduler_outputs.chunked_prefill_sequences,
                scheduler_outputs.chunked_prefill_tokens,
            )

        # 2) batched full prefill
        if scheduler_outputs.prefill_sequences:
            self._run_batched_prefill(
                scheduler_outputs.prefill_sequences
            )

        # 3) batched decode
        if scheduler_outputs.decode_sequences:
            self._run_batched_decode(
                scheduler_outputs.decode_sequences
            )

        # 4) finalize finished seqs
        newly_finished = self.scheduler.update_sequences(self.tokenizer.eos_token_id)

        for seq in newly_finished:
            completed_outputs.append(self._create_output(seq))
            if seq.block_table is not None:
                self.block_manager.free_sequence_blocks(seq.block_table)
                seq.block_table = None

        return completed_outputs

    def run_to_completion(self) -> List[GenerationOutput]:
        outputs = []
        while self.scheduler.has_pending_requests():
            outputs.extend(self.step())
        outputs.sort(key=lambda x: x.seq_id)
        return outputs

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.add_request(prompt, max_tokens=max_tokens)
        outputs = self.run_to_completion()
        return outputs[0].generated_text if outputs else ""

    def generate_batch(self, prompts: List[str], max_tokens: int = 100) -> List[str]:
        for prompt in prompts:
            self.add_request(prompt, max_tokens=max_tokens)
        outputs = self.run_to_completion()
        return [o.generated_text for o in outputs]

    # ------------------------------------------------
    # Prefill / chunked prefill / decode
    # ------------------------------------------------

    def _run_batched_prefill(self, sequences: List[Sequence]) -> None:
        """
        Full prompt prefill, batched.

        Model input:
          input_ids [B, T]
          metadata.seq_lens [B]
          metadata.slot_mapping [N]
        """
        token_batches: List[List[int]] = []
        seq_lens_list: List[int] = []

        # allocate blocks up front and collect prompt tokens
        for seq in sequences:
            prompt_len = seq.get_prompt_len()
            num_blocks_needed = compute_num_blocks(prompt_len, self.block_size)

            if seq.block_table is None:
                seq.block_table = self.block_manager.allocate_blocks_for_sequence(num_blocks_needed)

            token_batches.append(seq.prompt_token_ids)
            seq_lens_list.append(prompt_len)

        input_ids, seq_lens = pad_2d_int(
            token_batches,
            pad_value=self.tokenizer.pad_token_id,
            device=self.device,
        )

        slot_mapping = self._build_slot_mapping_for_prefill(sequences, seq_lens_list)

        metadata = AttentionMetadata(
            is_prefill=True,
            block_tables=None,                # not needed for flash prefill
            context_lens=None,                # not needed for flash prefill
            seq_lens=seq_lens,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
        )

        logits = self.model(
            input_ids=input_ids,
            metadata=metadata,
            kv_cache=self.block_kv_cache,
        )

        # sample 1 token per sequence using the last valid prefill position
        next_tokens = self._sample_from_prefill_logits(logits, seq_lens)

        for i, seq in enumerate(sequences):
            seq.num_prefilled_tokens = seq.get_prompt_len()
            seq.append_token(next_tokens[i].item())

    def _run_batched_chunked_prefill(self, sequences: List[Sequence], num_tokens_per_seq: List[int]) -> None:
        """
        Batched chunked prefill.
        Each sequence contributes a chunk [start:end] from its prompt.
        """
        chunk_batches: List[List[int]] = []
        chunk_lens_list: List[int] = []
        total_context_after_chunk: List[int] = []

        # ensure enough blocks allocated for each chunk endpoint
        for seq, num_tokens in zip(sequences, num_tokens_per_seq):
            start = seq.num_prefilled_tokens
            end = start + num_tokens
            chunk = seq.prompt_token_ids[start:end]

            total_tokens_after = end
            blocks_needed = compute_num_blocks(total_tokens_after, self.block_size)

            if seq.block_table is None:
                seq.block_table = self.block_manager.allocate_blocks_for_sequence(blocks_needed)
            else:
                current_blocks = seq.block_table.num_blocks()
                extra = blocks_needed - current_blocks
                for _ in range(extra):
                    block_id = self.block_manager.allocate_block()
                    seq.block_table.append_block(block_id)

            chunk_batches.append(chunk)
            chunk_lens_list.append(len(chunk))
            total_context_after_chunk.append(end)

        input_ids, seq_lens = pad_2d_int(
            chunk_batches,
            pad_value=self.tokenizer.pad_token_id,
            device=self.device,
        )

        slot_mapping = self._build_slot_mapping_for_chunked_prefill(sequences, chunk_lens_list)

        metadata = AttentionMetadata(
            is_prefill=True,
            block_tables=None,
            context_lens=None,
            seq_lens=seq_lens,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
        )

        logits = self.model(
            input_ids=input_ids,
            metadata=metadata,
            kv_cache=self.block_kv_cache,
        )

        # update progress
        finished_chunk_prefill_indices = []
        for i, (seq, num_tokens) in enumerate(zip(sequences, num_tokens_per_seq)):
            seq.num_prefilled_tokens += num_tokens
            if seq.num_prefilled_tokens >= seq.get_prompt_len():
                finished_chunk_prefill_indices.append(i)

        # sample first generated token only for sequences that just completed prefill
        if finished_chunk_prefill_indices:
            next_tokens = self._sample_from_prefill_logits(logits, seq_lens)
            for i in finished_chunk_prefill_indices:
                sequences[i].append_token(next_tokens[i].item())

    def _run_batched_decode(self, sequences: List[Sequence]) -> None:
        """
        Batched decode:
          input_ids [B, 1]
          block_tables [B, max_blocks]
          context_lens [B]
          slot_mapping [B]
        """
        # allocate block for next token if needed
        for seq in sequences:
            total_len_after_write = seq.get_len()
            blocks_needed = compute_num_blocks(total_len_after_write, self.block_size)

            if seq.block_table is None:
                seq.block_table = self.block_manager.allocate_blocks_for_sequence(blocks_needed)
            else:
                current_blocks = seq.block_table.num_blocks()
                extra = blocks_needed - current_blocks
                for _ in range(extra):
                    block_id = self.block_manager.allocate_block()
                    seq.block_table.append_block(block_id)

        input_ids = torch.tensor(
            [[seq.get_last_token_id()] for seq in sequences],
            dtype=torch.long,
            device=self.device,
        )  # [B, 1]

        block_tables = pad_block_tables(
            [self._block_table_to_list(seq.block_table) for seq in sequences],
            device=self.device,
        )

        # context length here is the length INCLUDING the token being written/read for this step,
        # matching your decode kernel usage
        context_lens = torch.tensor(
            [seq.get_len() for seq in sequences],
            dtype=torch.int32,
            device=self.device,
        )

        slot_mapping = self._build_slot_mapping_for_decode(sequences)

        metadata = AttentionMetadata(
            is_prefill=False,
            block_tables=block_tables,
            context_lens=context_lens,
            seq_lens=None,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
        )

        logits = self.model(
            input_ids=input_ids,
            metadata=metadata,
            kv_cache=self.block_kv_cache,
        )

        next_tokens = self.sampler.sample(logits)  # [B]

        for i, seq in enumerate(sequences):
            seq.append_token(next_tokens[i].item())

    # ------------------------------------------------
    # Slot mapping builders
    # ------------------------------------------------

    def _build_slot_mapping_for_prefill(self, sequences: List[Sequence], seq_lens: List[int]) -> List[int]:
        """
        Flatten order must match flatten_valid_kv:
          sequence-major, then token-major within sequence.
        """
        slots: List[int] = []
        for seq, L in zip(sequences, seq_lens):
            for token_pos in range(L):
                slots.append(self._logical_pos_to_slot(seq, token_pos))
        return slots

    def _build_slot_mapping_for_chunked_prefill(self, sequences: List[Sequence], chunk_lens: List[int]) -> List[int]:
        """
        Flatten order:
          sequence-major, then chunk token order within each sequence
        """
        slots: List[int] = []
        for seq, chunk_len in zip(sequences, chunk_lens):
            start = seq.num_prefilled_tokens
            for local_idx in range(chunk_len):
                logical_pos = start + local_idx
                slots.append(self._logical_pos_to_slot(seq, logical_pos))
        return slots

    def _build_slot_mapping_for_decode(self, sequences: List[Sequence]) -> List[int]:
        """
        One new token written per sequence in decode.
        The token being written lands at logical position seq.get_len() - 1
        under your current decode convention.
        """
        slots: List[int] = []
        for seq in sequences:
            logical_pos = seq.get_len() - 1
            slots.append(self._logical_pos_to_slot(seq, logical_pos))
        return slots

    def _logical_pos_to_slot(self, seq: Sequence, logical_pos: int) -> int:
        """
        Convert logical token position -> flattened slot_id
        slot_id = physical_block_id * block_size + offset_in_block
        """
        block_idx = logical_pos // self.block_size
        offset_in_block = logical_pos % self.block_size

        physical_block_ids = self._block_table_to_list(seq.block_table)
        physical_block_id = physical_block_ids[block_idx]

        return physical_block_id * self.block_size + offset_in_block

    def _block_table_to_list(self, block_table) -> List[int]:
        """
        Adapt this to your actual BlockTable API.
        """
        if hasattr(block_table, "physical_block_ids"):
            return list(block_table.physical_block_ids())
        if hasattr(block_table, "block_ids"):
            return list(block_table.block_ids)
        if isinstance(block_table, list):
            return block_table
        raise TypeError("Unsupported block_table format; adapt _block_table_to_list().")

    # ------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------

    def _sample_from_prefill_logits(self, logits: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """
        logits assumed [B, T, vocab]
        sample from last valid token of each sequence
        """
        B = logits.shape[0]
        last_logits = []
        for i in range(B):
            last_idx = int(seq_lens[i].item()) - 1
            last_logits.append(logits[i, last_idx])
        last_logits = torch.stack(last_logits, dim=0)  # [B, vocab]
        return self.sampler.sample(last_logits)

    # ------------------------------------------------
    # Output helpers
    # ------------------------------------------------

    def _create_output(self, seq: Sequence) -> GenerationOutput:
        all_tokens = seq.get_token_ids()
        generated_text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)

        return GenerationOutput(
            seq_id=seq.seq_id,
            prompt=self._prompts.get(seq.seq_id, ""),
            generated_text=generated_text,
            prompt_tokens=seq.get_prompt_len(),
            generated_tokens=seq.get_output_len(),
        )

    def get_stats(self) -> dict:
        return {
            "model_layers": self.config.num_hidden_layers,
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "num_attention_heads": self.config.num_attention_heads,
            "num_kv_heads": self.config.num_key_value_heads,
            "max_seq_len": self.max_seq_len,
            "max_batch_size": self.max_batch_size,
            "block_size": self.block_size,
            "kv_cache_memory_mb": self.block_kv_cache.memory_usage_mb,
            "scheduler": str(self.scheduler),
        }