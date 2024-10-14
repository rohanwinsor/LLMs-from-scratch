import torch
import numpy as np
from tqdm import trange


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            print(eos_id)
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_new_text(idx: torch.Tensor, model, context_length, max_output_token):
    for _ in trange(max_output_token):
        idx_inp = idx[:, -context_length:]
        logits = model(idx_inp)
        # (batch, T, out_dim) -> (batch, out_dim) of final element in seq
        logits = logits[:, -1, :]
        # proba = torch.softmax(logits, dim=-1)
        out_tok = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.concat((idx, out_tok), dim=1)
    return idx


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, " "Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_embed.weight = assign(gpt.pos_embed.weight, params["wpe"])
    gpt.tok_embed.weight = assign(gpt.tok_embed.weight, params["wte"])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.transformer_blocks[b].mhma.W_queries.weight = assign(
            gpt.transformer_blocks[b].mhma.W_queries.weight, q_w.T
        )
        gpt.transformer_blocks[b].mhma.W_keys.weight = assign(
            gpt.transformer_blocks[b].mhma.W_keys.weight, k_w.T
        )
        gpt.transformer_blocks[b].mhma.W_values.weight = assign(
            gpt.transformer_blocks[b].mhma.W_values.weight, v_w.T
        )
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.transformer_blocks[b].mhma.W_queries.bias = assign(
            gpt.transformer_blocks[b].mhma.W_queries.bias, q_b
        )
        gpt.transformer_blocks[b].mhma.W_keys.bias = assign(
            gpt.transformer_blocks[b].mhma.W_keys.bias, k_b
        )
        gpt.transformer_blocks[b].mhma.W_values.bias = assign(
            gpt.transformer_blocks[b].mhma.W_values.bias, v_b
        )
        gpt.transformer_blocks[b].mhma.out_proj.weight = assign(
            gpt.transformer_blocks[b].mhma.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[b].mhma.out_proj.bias = assign(
            gpt.transformer_blocks[b].mhma.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )
        gpt.transformer_blocks[b].ff.layers[0].weight = assign(
            gpt.transformer_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.transformer_blocks[b].ff.layers[0].bias = assign(
            gpt.transformer_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )
        gpt.transformer_blocks[b].ff.layers[2].weight = assign(
            gpt.transformer_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[b].ff.layers[2].bias = assign(
            gpt.transformer_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )
        gpt.transformer_blocks[b].layer_norm1.scale = assign(
            gpt.transformer_blocks[b].layer_norm1.scale,
            params["blocks"][b]["ln_1"]["g"],
        )
        gpt.transformer_blocks[b].layer_norm1.shift = assign(
            gpt.transformer_blocks[b].layer_norm1.shift,
            params["blocks"][b]["ln_1"]["b"],
        )
        gpt.transformer_blocks[b].layer_norm2.scale = assign(
            gpt.transformer_blocks[b].layer_norm2.scale,
            params["blocks"][b]["ln_2"]["g"],
        )
        gpt.transformer_blocks[b].layer_norm2.shift = assign(
            gpt.transformer_blocks[b].layer_norm2.shift,
            params["blocks"][b]["ln_2"]["b"],
        )
    gpt.layer_norm.scale = assign(gpt.layer_norm.scale, params["g"])
    gpt.layer_norm.shift = assign(gpt.layer_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
