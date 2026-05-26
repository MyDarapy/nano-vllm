"""sampling strategies for text generation"""
import torch 

class Sampler:
    def __init__(self, k=50, t=1.0):
        self.temperature = t
        self.k = k 

    def greedy_decoding(self, logits):
        # Get logits for the last position only
        #logits.shape [batch_size, seq_len, vocab_size]

        last_logits = logits[:, -1, :]
        next_tokens = torch.argmax(last_logits, dim=-1) 

        return next_tokens
    
    def top_k_sampling(self, logits):
        pass

    def top_p_sampling(self, logits):
        pass
