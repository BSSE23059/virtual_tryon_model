import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricMatchingModule(nn.Module):
    def __init__(self, input_nc=6):
        super(GeometricMatchingModule, self).__init__()
        
        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_nc, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True)
        )
        
        # Regression head
        self.regression = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 2, 3, padding=1)  # 2 channels for x,y offset
        )
        
    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        features = self.feature_extraction(x)
        flow_field = self.regression(features)
        return flow_field