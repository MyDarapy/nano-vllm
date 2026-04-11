"""Block manager for pagedattention. 

Manages physical KV blocks allocation and deallocation per sequence  

Manages prefix caching """

from vllm.engine.core.block import Block, BLOCK_SIZE, BlockTable, compute_blocks, hash_token_block


class BlockManager:
    def __init__(self, block_size, num_blocks, enable_prefix_caching=True):
        self.num_blocks = num_blocks
        self.block_size = BLOCK_SIZE
        self.enable_prefix_caching = enable_prefix_caching

        self.free_blocks = list(range(num_blocks-1, -1, -1))

        # track block metadata for reference counting 
        for i in range(num_blocks):
            self.blocks = [Block(block_id=i, block_size=self.block_size)]

        self.prefix_cache = {}
        self._block_to_cache_key = {}

        
    def allocate_block(self):
        if not self.free_blocks:
            raise RuntimeError("No KV cache blocks to allocate!")
            
        block_id = self.free_blocks.pop()
        block = self.blocks[block_id]
        block.ref_count = 1
        block.prefix_cache = None
        block.is_full = False
        return block_id 
    
    def can_allocate(self, num_blocks):
        "Check if we have enough free blocks to allocate"
        check = len(self.free_blocks) >= num_blocks
        return check
    
    def can_allocate_seq_len(self, seq_len):
        "Check if we can allocate blocks for a sequence of given length"
        num_blocks = compute_blocks(seq_len, self.block_size)
        return self.can_allocate(num_blocks)
    
    def get_num_free_blocks(self):
        return len(self.free_blocks)

    def get_num_used_blocks(self):
        return int(self.num_blocks - len(self.free_blocks))
    
    def get_utilization(self):
        if self.num_blocks == 0:
            return 0.00
        return self.get_num_used_blocks /self.num_blocks
    
    def free_blocks(self, block_id):
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Invalid block_id {block_id}. Block does not exist. Cannot deallocate non existent block")
        block = self.blocks[block_id]
        new_ref_count = block.decrement_ref()

        if new_ref_count <0:
            raise RuntimeError("Ref count went negative - double free detected")

        
    def allocate_blocks_for_sequence(self, num_blocks):
        """Allocate new blocks for a sequence (no prefix cache sharing)"""
        if not self.can_allocate(num_blocks):
            raise RuntimeError(f"Cannot allocate {num_blocks} blocks. Only {self.get_num_free_blocks} is available."
                               f"KV cache has a total of {self.num_blocks} blocks. {self.get_num_used_blocks} of which is in use")
        
        block_table = BlockTable(self.block_size)
        for _ in range(num_blocks):
            block_id = self.allocate_block()
            block_table.append(block_id)
        
        return block_table
    
    def allocate_block_with_prefix_caching(self, token_ids):
        """Alocate new blocks for a sequence with prefix caching. 

        It tries to  look for cache prefix blocks and use those (reference) before allocating new ones"""

        if not self.enable_prefix_caching:
            num_blocks = compute_blocks(len(token_ids), self.block_size)
            return self.allocate_blocks_for_sequence(num_blocks), 0
        
        block_table = BlockTable(self.block_size)
        shared_prefix_len = 0 
        parent_hash = None

        num_tokens_in_seq = len(token_ids)
        num_blocks_for_seq = num_tokens_in_seq // self.block_size

        for block_idx in range(num_blocks_for_seq):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = tuple(token_ids[start:end])

            cache_key = (parent_hash, block_tokens)

            if cache_key in self.prefix_cache:
                # Cache hit! 
                cached_block_id = self.prefix_cache[cache_key]
                cached_block = self.blocks[cached_block_id]

                # Increment reference count since 
                cached_block.increment_ref()
                block_table.append_block(cached_block_id)
                shared_prefix_len += self.block_size

                #update the parent block for the next hash 
                parent_hash = cached_block.prefix_hash

            else:
                if not self.can_allocate(1):
                    raise RuntimeError("Out of free KV cache blocks")
                
                block_id = self.allocate_block()
                block = self.block[block_id]       
                block.prefix_hash = hash_token_block(block_tokens, parent_hash)
                block.is_full = True

                # add prefix tokens to cache 
                self.prefix_cache[cache_key] = block_id

                # reserve mapping for easily O(n) removal 
                self._block_to_cache_key[block_id] = cache_key

                block_table.append_block(block_id)
                parent_hash = block.prefix_hash

            # allocate remaining partial blocks
            if num_tokens_in_seq % self.block_size > 0:
                if not self.allocate_block(1):
                    raise RuntimeError("Cannot allocate new KV cache bloch. Out of free blocks to allocate")
                block_id = self.allocate_block()
                block_table.append_block(block_id)

        return block_table, shared_prefix_len
    

    def mark_block_full(self, block_id, token_ids, parent_hash):
        if len(token_ids) != self.block_size:
            raise ValueError(f"Block must have {self.block_size} tokens") 
        
        if not self.enable_prefix_caching:
            return 0
        

    def free_sequence_blocks(self, block_table):
        for block_id in block_table.block_ids:
            self.free_blocks(block_id)
            block_table.ids.clear()

    def get_prefix_cache_stat(self):
        num_cached_blocks = len(self.prefix_cache)
        for block_id in self.prefix_cache.values:
            total_ref = sum(self.blocks[block_id].ref_count)

        return {"numbr_of_cached_blocks": num_cached_blocks,
                "total_reference" : total_ref,
                "avg_reference_per_block" : total_ref / num_cached_blocks if num_cached_blocks > 0 else 0 
        }
    
                





        



