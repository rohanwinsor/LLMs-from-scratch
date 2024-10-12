import torch
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

    def forward(self, x):
        B, T, C = x.shape
        print(T)
        key = self.k_W(x)
        value = self.v_W(x)
        query = self.q_W(x)
        key = key.view(B, T, self.n_head, self.head_dim)
        value = value.view(B, T, self.n_head, self.head_dim)
        query = query.view(B, T, self.n_head, self.head_dim)
        # Transpose: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)
        """
        # WHY NOT THIS DIRECTLY
        key = key.view(B, self.head_dim, T, self.n_head)
        value = value.view(B, self.head_dim, T, self.n_head)
        query = query.view(B, self.head_dim, T, self.n_head)
        """
        omega: torch.Tensor = query @ key.transpose(2, 3)
        # mask omega
        omega.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)
        att_weight = torch.softmax(omega / key.shape[-1] ** 0.5, dim=-1)
        context_vec = (att_weight @ value).transpose(2, 3)
        context_vec = context_vec.contiguous().view(B, T, self.d_out)
        return context_vec


if __name__ == "__main__":
    torch.manual_seed(123)
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your (x^1)
            [0.55, 0.87, 0.66],  # journey (x^2)
            [0.57, 0.85, 0.64],  # starts (x^3)
            [0.22, 0.58, 0.33],  # with (x^4)
            [0.77, 0.25, 0.10],  # one (x^5)
            [0.05, 0.80, 0.55],
        ]
    )
    print(MultiHeadAttentionV0(3, 1, 6, False, 2)(torch.stack((inputs, inputs))).shape)
    print(MultiHeadAttention(3, 3, 6)(torch.stack((inputs, inputs))))
