import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_channels=6, output_channels=3, ngf=64):
        super(Generator, self).__init__()
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, ngf, kernel_size=7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )
        
        # Downsampling
        self.down1 = self._downsample(ngf, ngf*2)
        self.down2 = self._downsample(ngf*2, ngf*4)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(ngf*4) for _ in range(9)]
        )
        
        # Upsampling
        self.up1 = self._upsample(ngf*4, ngf*2)
        self.up2 = self._upsample(ngf*2, ngf)
        
        # Output convolution
        self.conv_out = nn.Sequential(
            nn.Conv2d(ngf, output_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def _downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True)
        )

    def _upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.conv_out(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)