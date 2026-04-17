import torch

def input_padding(sequences, pad_value, device):
    if not sequences:
        return (
            torch.empty((0,0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            )
    lengths = [len(x) for x in sequences]
    max_len = max(lengths)
    out = torch.full((len(sequences), max_len), pad_value, dtype= torch.long, device=device)
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return out, torch.tensor(lengths, dtype=torch.long, device=device)


        
def flatten_valid_kv():
    pass