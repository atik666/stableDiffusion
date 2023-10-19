import torch
from torch import nn
from torch.nn import functional as F
from attntion import selfAttention

class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = selfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (Batch_size, Features, Height, Width)

        residue = x
        
        n, c, h, w = x.shape

        # (Batch_size, Features, Height, Width) -> (Batch_size, Features, Height * Width)
        x = x.view(n, c, h * w)

        # (Batch_size, Features, Height * Width) -> (Batch_size, Height * Width, Features)
        x = x.transpose(-1, -2)

        # (Batch_size, Height * Width, Features) -> (Batch_size, Height * Width, Features)
        x = self.attention(x)

        # (Batch_size, Height * Width, Features) -> (Batch_size, Features, Height, Width)
        x = x.transpose(-1, -2)

        # (Batch_size, Features, Height, Width) -> (Batch_size, Features, Height, Width)
        x = x.view(n, c, h, w)

        x += residue

        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x: (Batch_size, In_channel, Height, Width)
        residue = x

        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)