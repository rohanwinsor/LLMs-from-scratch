import torch.nn as nn
import torch
import json
from utils.gpt_download import download_and_load_gpt2
from dataclasses import dataclass
from utils.utils import generate_new_text, load_weights_into_gpt


@dataclass
class GPTConfig:
    vocab_size: int = 50257  # Vocabulary size
    context_length: int = 1024  # Context length
    emb_dim: int = 768  # Embedding dimension
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    drop_rate: float = 0.1  # Dropout rate
    qkv_bias: bool = True  # Query-Key-Value bias


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_len, dropout, n_head, qkv_bias=False
    ) -> None:
        assert d_out % n_head == 0
        super().__init__()
        # Getting Model Dims
        self.d_out = d_out
        self.n_head = n_head
        self.head_dim = d_out // n_head
        # Init Trainable parameters
        self.W_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x):
        B, T, C = x.shape
        keys: torch.Tensor = self.W_keys(x).view(B, T, self.n_head, self.head_dim)
        queries: torch.Tensor = self.W_queries(x).view(B, T, self.n_head, self.head_dim)
        values: torch.Tensor = self.W_values(x).view(B, T, self.n_head, self.head_dim)
        # (batch_size, num_of_tokens, no_of_heads, head_dim) -> (batch_size, no_of_heads, num_of_tokens, head_dim)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        omega: torch.Tensor = queries @ keys.transpose(
            2, 3
        )  # (batch_size, no_of_heads, num_of_tokens, head_dim) -> (batch_size, no_of_heads, head_dim, num_of_tokens)
        # mask
        omega.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)
        # scaled omega or att_score with dropout
        att_weight: torch.Tensor = self.dropout(
            torch.softmax(omega / keys.shape[-1] ** 0.5, dim=-1)
        )
        # (batch_size, no_of_heads, num_of_tokens, head_dim) -> (batch_size, num_of_tokens, no_of_heads, head_dim)
        context: torch.Tensor = (att_weight @ values).transpose(1, 2)
        return self.out_proj(context.contiguous().view(B, T, self.d_out))


class TransformerBlock(nn.Module):
    # input -> LayerNorm2 -> MHA -> DropOut -> residual connection
    # -> LayerNorm2 -> FF -> Dropout -> residual connection -> output
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.mhma = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_len=cfg.context_length,
            dropout=cfg.drop_rate,
            n_head=cfg.n_heads,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg)
        self.layer_norm1 = LayerNorm(embed_dim=cfg.emb_dim)
        self.layer_norm2 = LayerNorm(embed_dim=cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.mhma(x)
        x = self.dropout(x)
        x = x + shortcut
        shortcut = x
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        return x + shortcut


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    """
    The main idea behind layer normalization is to adjust the activations (outputs)
    of a neural network layer to have a mean of 0 and a variance of 1
    """

    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.epislon = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.epislon)
        return self.scale * norm + self.shift


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # tok embed, pos embed, TransformerBlock, LayerNorm, Linear
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_embed = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_out = nn.Dropout(cfg.drop_rate)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.layer_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, inp_toks):
        B, T = inp_toks.shape
        tok_emb = self.tok_embed(inp_toks)
        pos_emb = self.pos_embed(torch.arange(T, device=inp_toks.device))
        # adding pos embedding to the tok embedding
        x = tok_emb + pos_emb
        x = self.drop_out(x)
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        logits = self.out_head(x)
        return logits


if __name__ == "__main__":
    import tiktoken
    from utils.utils import generate, text_to_token_ids, token_ids_to_text

    tokenizer = tiktoken.get_encoding("gpt2")
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch = torch.stack(
        [torch.tensor(tokenizer.encode(txt1)), torch.tensor(tokenizer.encode(txt2))],
        dim=0,
    )
    torch.manual_seed(123)
    gpt_config = GPTConfig()
    model = GPTModel(cfg=gpt_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)
    # ## Load Open AI Weights
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    model = GPTModel(gpt_config)
    load_weights_into_gpt(model, params)
    model.eval()
    logits = generate_new_text(batch, model, gpt_config.context_length, 10)
    print(tokenizer.decode(logits.tolist()[0]))
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to("cpu"),
        max_new_tokens=25,
        context_size=1000,
        top_k=50,
        temperature=1.5,
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
