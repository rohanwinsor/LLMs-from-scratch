import torch
import torch.nn as nn
import torch.functional as F


class SelfAttentationV0(nn.Module):
    def __init__(self, inp_dim, out_dim) -> None:
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(size=(inp_dim, out_dim)))
        self.W_key = nn.Parameter(torch.rand(size=(inp_dim, out_dim)))
        self.W_value = nn.Parameter(torch.rand(size=(inp_dim, out_dim)))

    def forward(self, x):
        key = x @ self.W_key
        value = x @ self.W_value
        query = x @ self.W_query
        att_score = query @ key.T  # omega
        # scaled dot product
        att_weight = torch.softmax(att_score / (key.shape[-1] ** 0.5), dim=-1)
        context_vec = att_weight @ value
        return context_vec


class SelfAttentationV1(nn.Module):
    def __init__(self, inp_dim, out_dim, qkv_bias=False) -> None:
        super().__init__()
        self.W_query = nn.Linear(inp_dim, out_dim, bias=qkv_bias)
        self.W_key = nn.Linear(inp_dim, out_dim, bias=qkv_bias)
        self.W_value = nn.Linear(inp_dim, out_dim, bias=qkv_bias)

    def forward(self, x):
        key = self.W_key(x)
        value = self.W_value(x)
        query = self.W_query(x)
        att_score = query @ key.T  # omega
        # scaled dot product
        att_weight = torch.softmax(att_score / (key.shape[-1] ** 0.5), dim=-1)
        context_vec = att_weight @ value
        return context_vec


class CausalAttentation(nn.Module):
    def __init__(self, d_in, d_out, context_len, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout()
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        ## Batching
        B, T, C = x.shape  # Batch Size, Sequence Size, Embedding Dim
        print(x.shape)
        self.keys = self.W_key(x)
        self.querys = self.W_query(x)
        self.value = self.W_value(x)
        omega: torch.Tensor = self.querys @ self.keys.transpose(1, 2) # batched transpose
        omega.masked_fill_(
            self.mask.bool()[:T, :T], -torch.inf
        )  # Could not understand this line
        # scale
        att_weight = self.dropout(torch.softmax(omega / self.keys.shape[-1] ** 0.5, dim=-1))
        context_vec = att_weight @ self.value
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
    model = SelfAttentationV0(3, 2)
    model2 = SelfAttentationV1(3, 2)
    model2.load_state_dict(model.state_dict(), strict=False)
    print(model(inputs))
    print(CausalAttentation(3, 2, 6)(torch.stack((inputs, inputs))))
