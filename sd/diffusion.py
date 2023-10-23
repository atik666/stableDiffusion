import torch
from torch.nn import nn
from torch.nn import functional as F
from attention import selfAttention, CrossAttention

class TimeEmbedding(nn.Module):

    def __init__(self, n_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)

        x = self.linear_1(x)

        x =  F.silu(x)

        x = self.linear_2(x)

        #(1, 1280)
        return x
    
class Upsample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch_size, Channels, Height, Width) -> (Batch_size, Channels, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # (Batch_size, Channels, Height * 2, Width * 2) -> (Batch_size, Channels, Height * 2, Width * 2)
        x = self.conv(x)

        return x


class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self: # type: ignore # TODO: Fix type ignore if it doesn't work
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    
class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders = nn.Module([
            # (Batch_size, 4, Height / 8, Width / 8) 
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (Batch_size, 320, Height / 8, Width / 8) -> (Batch_size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (Batch_size, 640, Height / 16, Width / 16) -> (Batch_size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch_size, 1280, Height / 32, Width / 32) -> (Batch_size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            # (Batch_size, 1280, Height / 64, Width / 64) -> (Batch_size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            
            UNET_AttentionBlock(8, 160),

            UNET_ResidualBlock(1280, 1280),
            )

        self.decoders = nn.ModuleList([
            # (Batch_size, 2560, Height / 64, Width / 64) -> (Batch_size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),  
        ])


class Diffusion(nn.Module):

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # latent: (Batch_size, 4, Height / 8, Width / 8)
        # context: (Batch_size, Seq_Len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch_size, 4, Height / 8, Width / 8) -> (Batch_size, 320, Height/8, Width / 8)
        output = self.unet(latent, context, time)

        # (Batch_size, 320, Height / 8, Width / 8) -> (Batch_size, 4, Height / 8, Width / 8)
        output = self.final(output)

        return output