import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNetPP(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        nb = base

        # Encoder path
        self.conv00 = ConvBlock(in_ch, nb)
        self.conv10 = ConvBlock(nb, nb * 2)
        self.conv20 = ConvBlock(nb * 2, nb * 4)
        self.conv30 = ConvBlock(nb * 4, nb * 8)
        self.conv40 = ConvBlock(nb * 8, nb * 16)

        # Decoder dense connections (corrected channel counts)
        self.up01 = ConvBlock(nb + nb * 2, nb)               # (64 + 128) = 192
        self.up11 = ConvBlock(nb * 2 + nb * 4, nb * 2)       # (128 + 256) = 384
        self.up21 = ConvBlock(nb * 4 + nb * 8, nb * 4)       # (256 + 512) = 768
        self.up31 = ConvBlock(nb * 8 + nb * 16, nb * 8)      # (512 + 1024) = 1536

        self.up02 = ConvBlock(nb * (1 + 1 + 2), nb)          # x00(64)+x01(64)+x11(128)=256
        self.up12 = ConvBlock(nb * (2 + 2 + 4), nb * 2)      # x10(128)+x11(128)+x21(256)=512
        self.up22 = ConvBlock(nb * (4 + 4 + 8), nb * 4)      # x20(256)+x21(256)+x31(512)=1024

        self.up03 = ConvBlock(nb * (1 + 1 + 1 + 2), nb)      # x00,x01,x02,x12upsampled=64+64+64+128=320
        self.up13 = ConvBlock(nb * (2 + 2 + 2 + 4), nb * 2)  # 128+128+128+256=640

        self.up04 = ConvBlock(nb * (1 + 1 + 1 + 1 + 2), nb)  # 64+64+64+64+128=384

        # Final 1x1 conv
        self.final = nn.Conv2d(nb, out_ch, 1)

    def upsample(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=True)

    def forward(self, x):
        # Encoder
        x00 = self.conv00(x)
        x10 = self.conv10(F.max_pool2d(x00, 2))
        x20 = self.conv20(F.max_pool2d(x10, 2))
        x30 = self.conv30(F.max_pool2d(x20, 2))
        x40 = self.conv40(F.max_pool2d(x30, 2))

        # Decoder dense connections
        x01 = self.up01(torch.cat([x00, self.upsample(x10, x00)], 1))
        x11 = self.up11(torch.cat([x10, self.upsample(x20, x10)], 1))
        x21 = self.up21(torch.cat([x20, self.upsample(x30, x20)], 1))
        x31 = self.up31(torch.cat([x30, self.upsample(x40, x30)], 1))

        x02 = self.up02(torch.cat([x00, x01, self.upsample(x11, x00)], 1))
        x12 = self.up12(torch.cat([x10, x11, self.upsample(x21, x10)], 1))
        x22 = self.up22(torch.cat([x20, x21, self.upsample(x31, x20)], 1))

        x03 = self.up03(torch.cat([x00, x01, x02, self.upsample(x12, x00)], 1))
        x13 = self.up13(torch.cat([x10, x11, x12, self.upsample(x22, x10)], 1))

        x04 = self.up04(torch.cat([x00, x01, x02, x03, self.upsample(x13, x00)], 1))

        return self.final(x04)
