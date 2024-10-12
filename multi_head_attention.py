import torch
from torch import Tensor
import torch.nn as nn
from attention import CausalAttention


class MultiHeadAttentionV0(nn.Module):
    def __init__(self, d_in, d_out, context_length, qkv_bias, n_head):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, qkv_bias)
                for _ in range(n_head)
            ]
        )

    def forward(self, x):
        return torch.concat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, n_head=3, qkv_bias=False):
        super().__init__()
        assert d_out % n_head == 0, Exception("d_out should be divisible by n_head")
        self.d_out = d_out
        self.n_head = n_head
        self.head_dim = d_out // self.n_head
        self.k_W = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.q_W = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.v_W = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        key: Tensor = self.k_W(x)
        value = self.v_W(x)
        query = self.q_W(x)
        # Transpose: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        key: Tensor = key.view(*x.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        value: Tensor = value.view(*x.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        query: Tensor = query.view(*x.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        omega: Tensor = query @ key.transpose(2, 3)
        # mask omega
        omega.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)
        att_weight = torch.softmax(omega / key.shape[-1] ** 0.5, dim=-1)
        context_vec = (
            (att_weight @ value).transpose(2, 3).contiguous().view(B, T, self.d_out)
        )  # re-assemble all head outputs side by side, still don't understand though
        return context_vec


class MultiHeadAttentionNoBatch(nn.Module):
    def __init__(self, d_in, d_out, context_len, n_head=3, qkv_bias=False):
        super().__init__()
        assert d_out % n_head == 0
        self.head_dim = d_out // n_head
        self.n_head = n_head
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x):
        T, C = x.shape
        print(T)
        # if you don't do it like this chatgpt told it will act like channel
        # makes sense but give me source bitch
        # Transpose: (num_tokens, num_heads, head_dim) -> (num_heads, num_tokens, head_dim)
        keys = self.W_k(x).view(T, self.n_head, self.head_dim).permute(1, 0, 2)
        queries = self.W_q(x).view(T, self.n_head, self.head_dim).permute(1, 0, 2)
        values = self.W_v(x).view(T, self.n_head, self.head_dim).permute(1, 0, 2)
        omega = queries @ keys.transpose(1, 2)
        omega.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)
        att_w = torch.softmax(omega / keys.shape[-1] ** -0.5, dim=-1)
        # (num_tokens, num_heads, head_dim)
        context = (att_w @ values).transpose(1, 2).contiguous().view(T, self.d_out)
        return context


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
    print(
        MultiHeadAttentionV0(3, 3, 6, False, 3)(inputs.view(1, *inputs.shape))
        .view(6, 9)
        .shape
    )
    print(MultiHeadAttention(3, 9, 6)(inputs.view(1, *inputs.shape)).view(6, 9).shape)
    print(MultiHeadAttentionNoBatch(3, 9, 6)(inputs).shape)
