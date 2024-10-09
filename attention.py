import torch
import torch.nn as nn
import torch.functional as F


class SelfAttentionV0(nn.Module):
    def __init__(self, inp_dim, out_dim) -> None:
        super().__init__()
        self.W_key = nn.Parameter(torch.rand(size=(inp_dim, out_dim)))
        self.W_value = nn.Parameter(torch.rand(size=(inp_dim, out_dim)))
        self.W_query = nn.Parameter(torch.rand(size=(inp_dim, out_dim)))

    def forward(self, x):
        key = x @ self.W_key
        value = x @ self.W_value
        query = x @ self.W_query
        att_score = query @ key.T  # omega
        # scaled dot product
        att_weight = torch.softmax(att_score / (key.shape[-1] ** 0.5), dim=-1)
        context_vec = att_weight @ value
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
    sa_v0 = SelfAttentionV0(3, 2)
    print(sa_v0(inputs))
