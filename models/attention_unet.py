import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 1️⃣ 공간 크기가 다를 경우 보간해서 맞춤
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 2️⃣ 채널 축 병합
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # 3️⃣ Attention Mask 적용
        return x * psi



class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_c=64):
        super().__init__()
        self.inc = ConvBlock(in_ch, base_c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c, base_c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c * 2, base_c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c * 4, base_c * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c * 8, base_c * 16))

        # Attention blocks (fixed channel sizes)
        self.att1 = AttentionBlock(F_g=base_c * 16, F_l=base_c * 8, F_int=base_c * 4)
        self.att2 = AttentionBlock(F_g=base_c * 8, F_l=base_c * 4, F_int=base_c * 2)
        self.att3 = AttentionBlock(F_g=base_c * 4, F_l=base_c * 2, F_int=base_c)
        self.att4 = AttentionBlock(F_g=base_c * 2, F_l=base_c, F_int=base_c // 2)

        self.up1 = UpBlock(base_c * 16, base_c * 8)
        self.up2 = UpBlock(base_c * 8, base_c * 4)
        self.up3 = UpBlock(base_c * 4, base_c * 2)
        self.up4 = UpBlock(base_c * 2, base_c)

        self.outc = nn.Conv2d(base_c, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Attention gates
        x4 = self.att1(x5, x4)
        x3 = self.att2(x4, x3)
        x2 = self.att3(x3, x2)
        x1 = self.att4(x2, x1)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)
