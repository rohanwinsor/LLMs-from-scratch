import torch.nn as nn
import torch


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


if __name__ == "__main__":
    torch.manual_seed(123)
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your (x^1)
            [0.55, 0.87, 0.66],  # journey (x^2)
            [0.57, 0.85, 0.64],  # starts (x^3)
            [0.22, 0.58, 0.33],  # with (x^4)
            [0.77, 0.25, 0.10],  # one (x^5)
            [0.05, 0.80, 0.55],  # step (x^6)
        ]
    )
    mha = MultiHeadAttention(3, 2, 6, 0.0, 2)
    print(mha(torch.stack((inputs,))))
