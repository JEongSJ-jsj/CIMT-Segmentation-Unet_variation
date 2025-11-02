import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth_rate, layers):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_ch
        for _ in range(layers):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, growth_rate, 3, padding=1, bias=False)
            ))
            ch += growth_rate

    def forward(self, x):
        for layer in self.blocks:
            out = layer(x)
            x = torch.cat([x, out], 1)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.trans = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2)
        )
    def forward(self, x): return self.trans(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class DenseUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=36, growth=20, layers=6):
        """
        DenseUNet (≈9M parameters)
        Multi-stage upsampling for 256x256 output
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base, 7, padding=3)

        self.block1 = DenseBlock(base, growth, layers)
        self.trans1 = TransitionDown(base + growth * layers, base * 2)

        self.block2 = DenseBlock(base * 2, growth, layers)
        self.trans2 = TransitionDown(base * 2 + growth * layers, base * 4)

        self.block3 = DenseBlock(base * 4, growth, layers)
        self.trans3 = TransitionDown(base * 4 + growth * layers, base * 8)

        self.block4 = DenseBlock(base * 8, growth, layers)

        # Upsampling path
        self.up1 = UpBlock(base * 8 + growth * layers, base * 4)
        self.up2 = UpBlock(base * 4, base * 2)
        self.up3 = UpBlock(base * 2, base)

        self.final = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x); x = self.trans1(x)
        x = self.block2(x); x = self.trans2(x)
        x = self.block3(x); x = self.trans3(x)
        x = self.block4(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.final(x)
