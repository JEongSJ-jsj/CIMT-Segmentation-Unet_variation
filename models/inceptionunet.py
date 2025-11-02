import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Inception-style convolution block
# ----------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch1 = nn.Conv2d(in_ch, out_ch // 4, kernel_size=1)

        branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, kernel_size=1),
            nn.Conv2d(out_ch // 4, out_ch // 4, kernel_size=3, padding=1)
        )

        branch5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, kernel_size=1),
            nn.Conv2d(out_ch // 4, out_ch // 4, kernel_size=5, padding=2)
        )

        branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, out_ch // 4, kernel_size=1)
        )

        self.branches = nn.ModuleList([branch1, branch3, branch5, branch_pool])
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        outputs = [b(x) for b in self.branches]
        x = torch.cat(outputs, dim=1)
        return self.relu(self.bn(x))


# ----------------------------
# Encoder & Decoder blocks
# ----------------------------
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = InceptionBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = InceptionBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Align spatial dims
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ----------------------------
# Full InceptionUNet
# ----------------------------
class InceptionUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=60):
        """
        Inception-UNet
        Output: same spatial size as input (256×256)
        Parameters ≈ 7.9M
        """
        super().__init__()
        self.inc = InceptionBlock(in_ch, base)
        self.down1 = DownBlock(base, base * 2)
        self.down2 = DownBlock(base * 2, base * 4)
        self.down3 = DownBlock(base * 4, base * 8)
        self.down4 = DownBlock(base * 8, base * 16)

        self.up1 = UpBlock(base * 16, base * 8)
        self.up2 = UpBlock(base * 8, base * 4)
        self.up3 = UpBlock(base * 4, base * 2)
        self.up4 = UpBlock(base * 2, base)  # ✅ restored, fixed channels

        self.outc = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)  # ✅ restored, now matches 256×256
        return self.outc(x)
