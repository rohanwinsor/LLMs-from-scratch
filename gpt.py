import torch.nn as nn
import torch
import tiktoken
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257  # Vocabulary size
    context_length: int = 1024  # Context length
    emb_dim: int = 768  # Embedding dimension
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    drop_rate: float = 0.1  # Dropout rate
    qkv_bias: bool = False  # Query-Key-Value bias


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
        self.obj_proj = nn.Linear(d_out, d_out)
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
        return self.obj_proj(context.contiguous().view(B, T, self.d_out))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(nn.Module):
    """
    The main idea behind layer normalization is to adjust the activations (outputs)
    of a neural network layer to have a mean of 0 and a variance of 1
    """

    def __init__(self, embed_dim) -> None:
        super().__init__(embed_dim)
        self.epislon = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.epislon)
        return self.scale * norm + self.shift


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


class GPT2(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # tok embed, pos embed, TransformerBlock, LayerNorm, Linear
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_embed = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.layer_nodem = LayerNorm()
        self.drop_out = nn.Dropout(cfg.drop_rate)
        self.linear = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, inp_toks):
        B, T = inp_toks.shape
        tok_emb = self.tok_embed(inp_toks)
        pos_emb = self.pos_embed(torch.arange(T, device=inp_toks.device))
        # adding pos embedding to the tok embedding
        x = tok_emb + pos_emb
        x = self.drop_out(x)
        x = self.transformer_blocks(x)
        x = self.layer_nodem(x)
        return self.linear(x)


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch = torch.stack([torch.tensor(tokenizer.encode(txt1))], dim=0)
    gpt_config = GPTConfig()
    logits = GPT2(cfg=gpt_config)(batch)
    print(logits)
    print(logits.shape)
