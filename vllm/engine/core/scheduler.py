"""scheduler for continous batching"""

import heapq
from dataclasses import dataclass, field
from enum import Enum 


from vllm.engine.core.block import BLOCK_SIZE, compute_blocks
from vllm.engine.core.sequence import SequenceStatus, Sequence
from vllm.engine.core.block_manager import BlockManager


class SchedulingPolicy(Enum):
    FCFS = "fcfs" # order purely by arrival time
    PRIORITY = "priority"

@dataclass
class SchedulerOutputs:
    prefill_sequences = field(default_factory=list)
    decode_sequences = field(default_factory=list)
    preempted_sequences = field(default_factory=list)
    chunked_prefill_sequences = field(default_factory=list)
    chunked_prefill_tokens = field(default_factory=list)

    @property
    def num_prefill(self):
        return len(self.prefill_sequences)
    
    @property
    def num_decode(self):
        return len(self.decode_sequences)
    
    @property
    def num_preempted(self):
        return len(self.preempted_sequences)
    
    @property
    def num_chunked_prefill(self):
        return len(self.chunked_prefill_sequences)
    
    @property
    def total_sequnces(self):
        return self.num_prefill + self.num_decode = self.num_chunked_prefill


    def is_empty(self):
        return self.total_sequnces == 0
    

class Scheduler:
    def __init__(self, 
                 max_batch_size=8,
                 block_manager=None, 
                 block_size=BLOCK_SIZE,
                 scheduling_policy = SchedulingPolicy.PRIORITY,
                 enable_preemption=True,
                 max_prefill_tokens=512):
        
        self.max_batch_size = max_batch_size
        self.block_manager = block_manager
        self.block_size = block_size
        self.scheduling_policy = scheduling_policy
        self.enable_preeemption = enable_preemption
        self.max_prefill_tokens = max_prefill_tokens

        # In a min-heap highest priority is represented by a lower number
        self._waiting_heap = []
        self.running = []
        self.finished = []

        self._next_seq_id = 0

    def _get_priority_key(self, seq):
        if self.scheduling_policy == SchedulingPolicy.FCFS:
            # Ignore priority and only use arrival-time
            return (0, seq.arrival_time, seq.seq_id)
        else: 
            #higher priority first
            return (-seq.priority, seq.arrival_time, seq.seq_id)
    

    def _push_waiting(self, seq):
        key = self._get_priority_key(seq)
        heapq.heappush(self._waiting_heap, (key, seq)) # waiting_list/heap = [((key), seq)), ((key), seq))]

    def _pop_waiting(self):
        if not self._waiting_heap:
            return None
        _, seq = heapq.heappop(self._waiting_heap)
        return seq

    def _peek_waiting(self):
        if not self._waiting_heap:
            return None
        return self._waiting_heap[0][1]

    def waiting(self):
        sequence = []
        for _, seq in self._waiting_heap:
            sequence.append(seq)
        return sequence
    
    def add_request(
            self, 
            prompt_token_ids,
            max_tokens,
            priority=0):
        seq = Sequence(
            seq_id=self._next_seq_id, 
            prompt_token_ids=self.prompt_token_ids,
            status = SequenceStatus.WAITING,
            priority=priority,)
        
        self._next_seq_id += 1 
        self._push_waiting(seq)
        return seq
    
    def _handle_preemption(self, outputs):
        """Handle preemption of low-priority sequences for high priority waiting"""

        """preemeted sequences have their blocks freed and returned to waiting while the
        high priority tasks takes over"""

        if not self._waiting_heap or not self.running:
            return
        
        highest_waiting = self._peek_waiting()
        if highest_waiting is None:
            return 
        
        """If highest waiting can't get new blocks, we need to preempt lowest priority"""
        blocks_for_waiting = compute_blocks(highest_waiting.get_prompt_len(), self.block_size)

        while not self.block_mannager.can_allocate(blocks_for_waiting) and self.running:
            lowest_running = min(self.running, key=lambda s: (s.priority, -s.arrival_time))

            """Only preempt if waiting has higher priority"""
            if highest_waiting.priority <= lowest_running.priority:
                break 

            self.running.remove(lowest_running)
            outputs.preempted_sequences.append(lowest_running)
            
            # free blocks
            if lowest_running.block_table is not None:
                self.block_manager.free_sequence_


    def update_seuence(self, ):
        pass
    
    
    def schedule(self):
        outputs = 
        
        
