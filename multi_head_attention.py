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
    print(MultiHeadAttentionV0(3, 2, 6, False, 2)(torch.stack((inputs, inputs))))
