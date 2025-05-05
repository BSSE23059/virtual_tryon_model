import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, ndf=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input layer
            nn.Conv2d(input_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Hidden layers
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.main(x)