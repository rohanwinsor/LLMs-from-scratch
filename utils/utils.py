import torch
from tqdm import trange
def generate_new_text(idx: torch.Tensor, model, context_length, max_output_token):
    for _ in trange(max_output_token):
        idx_inp = idx[:, -context_length:]
        logits = model(idx_inp)
        # (batch, T, out_dim) -> (batch, out_dim) of final element in seq
        logits = logits[:, -1, :] 
        proba = torch.softmax(logits, dim=-1)
        out_tok = torch.argmax(proba, dim=-1, keepdim=True)
        idx = torch.concat((idx, out_tok), dim=1)
    return idx