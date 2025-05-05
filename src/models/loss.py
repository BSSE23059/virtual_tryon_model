import torch
import torch.nn as nn
import torchvision.models as models

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:29]
        self.model = nn.Sequential(*[vgg[i] for i in range(len(vgg))])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        return torch.mean((self.model(x) - self.model(y)) ** 2)

class GANLoss(nn.Module):
    def __init__(self, gan_mode='lsgan'):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
            
    def forward(self, prediction, target_is_real):
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)