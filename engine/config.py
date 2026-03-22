import os 
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class ModelConfig:
    model_path : str 
    vocab_size : int
    hidden_size : int
    intermidate_size : int
    num_hidden_layers : int 
    num_attention_heads : int 


    @classmethod
    def get_model_config(cls, model_path):
        hf_config = AutoConfig.from_pretrained(model_path)
        return cls (
            vocab_size = hf_config.vocab_size
            hidden_size = hf_config.hidden_size
            num_hidden_layers = hf_config.num_hidden_layers
        )

