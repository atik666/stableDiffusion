import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class selfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x: (Batch_size, Seq_len, Dim)

        input_shape = x.shape
        batch_size,  sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (Batch_size, Seq_len, Dim) -> (Batch_size, Seq_len, Dim * 3) -> 3 tensors of shape (Batch_size, Seq_len, Dim)
        q, k, v = torch.chunk(self.in_proj(x), 3, dim=-1)

        # (Batch_size, Seq_len, Dim) -> (Batch_size, Seq_len, H, Dim / H) -> (Batch_size, H, Seq_len, Dim / H)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # (Batch_size, H, Seq_len, Seq_len)
        weight = q @ k.transpose(-2, -1) 

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is made up of 1s
            mask = torch.ones_like(weight, dtype=torch.bool).triu(diagonal=1)
            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (Batch_size, H, Seq_len, Seq_len) @ (Batch_size, H, Seq_len, Dim / H) -> (Batch_size, H, Seq_len, Dim / H)
        output = weight @ v

        # (Batch_size, H, Seq_len, Dim / H) -> (Batch_size, Seq_len, H, Dim / H)
        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (Batch_size, Seq)
        return output
    
