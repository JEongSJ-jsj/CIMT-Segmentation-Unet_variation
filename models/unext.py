import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFFN(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim*expansion, 1),
            nn.GELU(),
            nn.Conv2d(dim*expansion, dim, 1)
        )
    def forward(self, x): return self.block(x)

class UNeXt(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64, expansion=4):
        super().__init__()
        self.enc1 = nn.Conv2d(in_ch, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base*2, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1)
        self.block = ConvFFN(base*4, expansion)
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.up2 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.outc = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.block(x3)
        x = self.up1(x4)
        x = self.up2(x)
        return self.outc(x)
