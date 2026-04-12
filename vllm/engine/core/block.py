from dataclasses import dataclass, field
import math
BLOCK_SIZE = 16 # NUMBER OF TOKENS PER BLOCK


def hash_token_block(token_ids, parent_hash):
    """" The hash token keeps track of similar blocks tokens e.g prefix caching.
    This is used for block sharing"""

    if parent_hash is None:
        return hash(token_ids)
    else:
        return hash (parent_hash, token_ids)
    

@dataclass
class Block:
    block_id : int
    block_size : int 
    ref_count: int = 1
    prefix_hash = None
    is_full = False

    def increment_ref(self):
        self.ref_count += 1

    def decrement_ref(self):
        self.ref_count -= 1
        return self.ref_count
    
    def __repr__(self):
        return f"Block_id: {self.block_id}, refs: {self.ref_count}, hash:{self.prefix_count}"

@dataclass
class BlockTable:
    # The index of this list represents the logical block index, the value represent the physical block
    block_ids = field(default_factory=list) 
    block_size = BLOCK_SIZE

    def get_block_id(self, logical_block_index):
        """logical block index is the index of block within a sequence. This function gets the physical block id
        for a given chunk of the sequence that sits in  logical block"""

        if logical_block_index < 0:
            raise IndexError(
                f"Invalid logical block index {logical_block_index}. Index must be non negative"
            )
        if logical_block_index >= len(self.block_ids):
            max_seq_len = len(self.block_ids) * self.block_size
            requested_token_range = (
                logical_block_index * self.block_size,
                (logical_block_index + 1) * self.block_size - 1
            )
            raise IndexError(
                f"Logical block {logical_block_index} not allocated"
                f"Table has a total of {len(self.block_ids)} blocks (max {max_seq_len} token)"
                f"Requested block covers token {requested_token_range[0]}-{requested_token_range[1]}. "
                f"Allocated blocks {self.block_ids}"

            )
        return self.block_ids[logical_block_index]
    
    def append_block(self, block_id):
        "Add newly allocated block when new tokens becomes available to the sequence"
        self.block_ids.append(block_id)

    def num_of_allocated_blocks(self):
        "The total number of physical block allocated so far for the sequence"
        return len(self.block_ids)
    
    def get_physical_block_id(self):
        return self.block_id.copy()
    
    def slot_mapping(self, seq_len):
        "slot_mapping[i] produces slot indices for exact posiitons where tokens KV are stored in global cache"
        "This is the logical step. The physical step then happens on the GPU to get the actual "
        "memory address where the slot is located w.r.t the whole cache "
        slots = []
        for pos in range(seq_len):
            logical_block = pos // self.block_size
            slot_in_block = pos % self.block_size
            physical_block = self.block_ids[logical_block]
            global_slot = physical_block  * self.block_size + slot_in_block
            slots.append(global_slot)
        return slots

def compute_blocks(seq_len, block_size=BLOCK_SIZE):
    total_blocks = math.ceil(seq_len / block_size)
    return total_blocks


