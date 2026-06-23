from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoConfig


@dataclass
class ModelConfig:
    model_path: str
    vocab_size : int
    hidden_size : int
    intermediate_size: int
    num_hidden_layers : int 
    num_attention_heads : int 
    num_kv_heads : int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def intermidate_size(self):
        return self.intermediate_size

    @property
    def intermidiate_size(self):
        return self.intermediate_size


    @classmethod
    def get_model_config(cls, model_path):
        hf_config = AutoConfig.from_pretrained(model_path)
        return cls (
            model_path=str(model_path),
            vocab_size = hf_config.vocab_size,
            hidden_size = hf_config.hidden_size,
            intermediate_size=getattr(hf_config, "intermediate_size"),
            num_hidden_layers = hf_config.num_hidden_layers,
            num_attention_heads = hf_config.num_attention_heads,
            num_kv_heads = hf_config.num_key_value_heads,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
        )


@dataclass
class Metadata:
    is_prefill : bool
    positions: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    context_lens : Optional[torch.Tensor] = None
    slot_mapping : Optional[torch.Tensor] = None
    num_sequences: Optional[int] = None
