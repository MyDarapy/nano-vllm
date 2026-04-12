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
    num_kv_heads : int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float

    @property
    def head_dim(self):
        self.hidden_size // self.num_attention_heads


    @classmethod
    def get_model_config(cls, model_path):
        hf_config = AutoConfig.from_pretrained(model_path)
        return cls (
            vocab_size = hf_config.vocab_size,
            hidden_size = hf_config.hidden_size,
            intermidate_size = hf_config.intermidate_size,
            num_hidden_layers = hf_config.num_hidden_layers,
            num_attention_heads = hf_config.num_attention_heads,
            num_kv_heads = hf_config.num_key_value_heads,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
        )

