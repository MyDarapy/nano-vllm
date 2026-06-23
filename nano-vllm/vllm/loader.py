from safetensors import safe_open
import torch 
from huggingface_hub import snapshot_download 
from  vllm.config import ModelConfig
from vllm.models.llama import LlamaForCausalLM
from pathlib import Path

def load_models(
        model_path,
        device="cuda",
        dtype= torch.float16,
        use_flash_attn = True,):
    
    # Download model if it is a Huggingface ID 
    local_path = _get_local_path(model_path)

    config = ModelConfig.get_model_config(local_path)
    print(f"Loading model: {model_path}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Layers: {config.num_hidden_layers}")
    print(f"Attention heads: {config.num_attention_heads}")
    print(f"  KV heads: {config.num_kv_heads}")
    print(f"  FlashAttention: {use_flash_attn}")

    model = LlamaForCausalLM(config)
    model.use_flash_attn = use_flash_attn
    state_dict = _load_weights(local_path)
    mapped_state_dict = _map_weights(state_dict)
    model.load_state_dict(mapped_state_dict, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model

def _get_local_path(model_path):
    path = Path(model_path)
    if path.exists():
        return path
    local_dir = snapshot_download(repo_id=model_path,
                                  allow_patterns=["*.safetensors", "*.json"],)
    
    return Path(local_dir)

def _load_weights(model_path):
    state_dict = {}

    safetensors_files = list(model_path.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")
    
    for filepath in safetensors_files:
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def _map_weights(hf_state_dict):
    mapped = {}
    for name, tensor in hf_state_dict.items():
        if "rotary_emb" in name:
            continue

        if name.startswith("model."):
            name = name[len("model."):]

        name = name.replace("self_attn.", "attention.")
        name = name.replace("input_layernorm.", "pre_layernorm.")
        name = name.replace("post_attention_layernorm.", "post_layernorm.")

        mapped[name] = tensor
    return mapped
